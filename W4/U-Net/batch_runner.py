# -*- coding: utf-8 -*-
"""批量推理引擎（与模型无关）：断点续跑 / 原子写 / 失败隔离 / 吞吐统计。

本文件在 U-Net 与 SegFormer 两个工程里**内容完全相同**，改一处记得同步另一处。
和模型相关的东西全在各自工程根目录的 batch_infer.py 里，以 Adapter 的形式提供：

    adapter.prep(path)                 -> {"x": CHW float32, "meta": {...}}
    adapter.infer(xs)                  -> [np.ndarray(h, w) uint8]   # xs: [N,C,H,W]
    adapter.post(mask_small, meta)     -> np.ndarray(H, W) uint8     # 裁填充 + 缩回原尺寸
    adapter.overlay(mask, path)        -> np.ndarray(H, W, 3) uint8
    adapter.colorize(mask)             -> np.ndarray(H, W, 3) uint8  # --mask-mode color 才用

为什么不用 glob
---------------
工程路径是 `D:\\[C]Remote_Sencing_Projection\\...`，方括号会被当成字符类，
`glob.glob` 会静默返回 0 个文件。这里一律用 os.listdir / os.walk。

断点续跑的判据是"结果完整"，不是"文件存在"
--------------------------------------------
  1) 每张图先写 mask 到 `<文件名>.tmp<pid>`，成功后 os.replace 原子改名；
  2) 改名成功之后，才往 manifest.csv 追加一行（这一行就是提交点）；
  3) 续跑时只有 manifest 里有行 **且** 产物文件存在且非空，才算这张做完了。

所以任何时刻 Ctrl-C 都不会留下"文件在、内容不全、还被当成功"的假完成：
最多是个没有 manifest 行的孤儿文件，下次重跑直接原子覆盖。
"""

import csv
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

__all__ = [
    "BatchRunner", "IMG_EXTS", "list_images", "build_keys", "sha1_of_file",
    "save_image_atomic", "save_json_atomic", "save_text_atomic", "fmt_dur", "now_str",
]

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")

MANIFEST_FIELDS = ["key", "src", "H", "W", "fg_ratio", "ms",
                   "mask", "overlay", "batch_id", "ts"]
FAILED_FIELDS = ["key", "src", "error", "attempts", "ts"]


# --------------------------------------------------------------------------- 小工具
def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fmt_dur(sec):
    """秒 -> 人类可读时长。"""
    sec = max(0.0, float(sec))
    if sec < 60:
        return "%.0fs" % sec
    if sec < 3600:
        return "%dm%02ds" % (sec // 60, sec % 60)
    return "%dh%02dm" % (sec // 3600, (sec % 3600) // 60)


def sha1_of_file(path, chunk=1 << 20):
    """权重指纹。用 sha1 而不是 mtime：mtime 会随拷贝/解压而变，sha1 只随内容变。"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write(path, writer):
    """先写 <path>.tmp<pid>，再 os.replace 原子改名。

    os.replace 在同一分区上是原子的——读者要么看到旧文件，要么看到完整的新文件，
    不会读到写了一半的内容。这是"续跑不会误判为完成"的底层保证。
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = "%s.tmp%d" % (path, os.getpid())
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def save_image_atomic(path, arr):
    """存 uint8 图像；按扩展名决定 TIFF 还是 PNG。"""
    ext = os.path.splitext(path)[1].lower()

    def writer(tmp):
        img = Image.fromarray(np.asarray(arr, dtype=np.uint8))
        if ext in (".tif", ".tiff"):
            img.save(tmp, format="TIFF")
        else:
            img.save(tmp, format="PNG")

    _atomic_write(path, writer)


def save_json_atomic(path, obj):
    def writer(tmp):
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    _atomic_write(path, writer)


def save_text_atomic(path, text):
    def writer(tmp):
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)

    _atomic_write(path, writer)


# --------------------------------------------------------------------------- 输入枚举
def _walk_images(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()                      # 保证顺序稳定，便于对比两次运行
        for name in sorted(filenames):
            if name.lower().endswith(IMG_EXTS):
                out.append(os.path.join(dirpath, name))
    return out


def list_images(inputs):
    """输入可以是：目录 / 图片文件 / 文本清单（每行一个路径，# 开头为注释）。

    返回绝对路径列表，保序去重。
    """
    found = []
    for p in inputs:
        if os.path.isdir(p):
            found.extend(_walk_images(p))
        elif os.path.isfile(p):
            if p.lower().endswith((".txt", ".lst")):
                with open(p, "r", encoding="utf-8-sig") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            found.append(line)
            else:
                found.append(p)
        else:
            raise SystemExit("[输入] 路径不存在：%s" % p)

    out, seen = [], set()
    for p in found:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


def build_keys(srcs):
    """给每张图一个稳定的主键（即输出文件名）。

    默认用文件名主干（stem）。**不同子目录里可能重名**（WHU 的 train/test 就重名），
    所以重名时自动拼上 8 位路径哈希，避免互相覆盖；这比"按顺序编号"更耐得住增删。
    """
    stems = [os.path.splitext(os.path.basename(s))[0] for s in srcs]
    counter = Counter(stems)
    keys, used = [], set()
    for src, stem in zip(srcs, stems):
        key = stem
        if counter[stem] > 1:
            digest = hashlib.md5(os.path.abspath(src).encode("utf-8")).hexdigest()[:8]
            key = "%s__%s" % (stem, digest)
        while key in used:                    # 理论上不会发生，兜一下
            key = key + "_"
        used.add(key)
        keys.append(key)
    return keys


# --------------------------------------------------------------------------- 引擎
class BatchRunner(object):
    """批量推理 + 断点续跑。

    输出目录结构（--out）：

        run_config.json     本次任务指纹（权重 sha1、输入尺寸、类别数…），续跑前校验
        manifest.csv        每张成功图一行：key/src/H/W/fg_ratio/ms/mask/overlay/ts
        failed.csv          当前仍失败的图 + 尝试次数 + 最后一次错误
        summary.json        本次运行的汇总
        masks/              <key>_mask.tif      uint8 索引图（--mask-mode color 时为彩色）
        overlays/           <key>_overlay.tif   原图 + 掩膜叠加
    """

    def __init__(self, out_dir, adapter, batch_size=8, prefetch=4,
                 resume=True, retry_failed=False, force_continue=False,
                 save_mask=True, save_overlay=True, mask_mode="index",
                 mask_ext=".tif", alpha=0.5, log_every=20, verbose=True):
        self.out = os.path.abspath(out_dir)
        self.adapter = adapter
        self.batch_size = max(1, int(batch_size))
        self.prefetch = max(1, int(prefetch))
        self.resume = bool(resume)
        self.retry_failed = bool(retry_failed)
        self.force_continue = bool(force_continue)
        self.save_mask = bool(save_mask)
        self.save_overlay = bool(save_overlay)
        self.mask_mode = mask_mode
        self.mask_ext = mask_ext if mask_ext.startswith(".") else "." + mask_ext
        self.alpha = float(alpha)
        self.log_every = max(1, int(log_every))
        self.verbose = verbose

        self.dir_mask = os.path.join(self.out, "masks")
        self.dir_overlay = os.path.join(self.out, "overlays")
        self.manifest_path = os.path.join(self.out, "manifest.csv")
        self.failed_path = os.path.join(self.out, "failed.csv")
        self.cfg_path = os.path.join(self.out, "run_config.json")

        self.done = {}
        self.failed = {}
        self.stats = dict(total=0, done=0, skipped=0, failed=0)
        self._pool = None

    # ------------------------------------------------------------------ 准备
    def prepare(self, signature):
        """建目录、读既有进度、校验任务指纹。signature 只放可 JSON 化的标量。"""
        os.makedirs(self.out, exist_ok=True)
        if self.save_mask:
            os.makedirs(self.dir_mask, exist_ok=True)
        if self.save_overlay:
            os.makedirs(self.dir_overlay, exist_ok=True)

        self.done = self._read_done() if self.resume else {}
        self.failed = self._read_failed() if self.resume else {}

        old = {}
        if self.resume and os.path.exists(self.cfg_path):
            try:
                with open(self.cfg_path, "r", encoding="utf-8") as f:
                    old = json.load(f) or {}
            except Exception:
                old = {}

        diff = []
        for k in sorted(signature):
            if k in old and old[k] != signature[k]:
                diff.append("    %-14s 已有 %r  ->  本次 %r" % (k, old[k], signature[k]))
        if diff and not self.force_continue:
            raise SystemExit(
                "[续跑] 输出目录里已经有一份参数不同的任务，拒绝混着跑：\n"
                + "\n".join(diff)
                + "\n  -> 换个 --out 目录；确认要混跑就加 --force-continue。")

        if diff:
            print("[续跑][警告] 参数与已有记录不一致，按 --force-continue 继续：")
            print("\n".join(diff))

        cfg = dict(signature)
        cfg["created"] = old.get("created", now_str())
        cfg["updated"] = now_str()
        save_json_atomic(self.cfg_path, cfg)

        print("[续跑] 输出目录 %s" % self.out)
        print("[续跑] 已完成 %d 张（manifest 有行且产物完整）| 上次失败 %d 张"
              % (len(self.done), len(self.failed)))
        return self

    # ------------------------------------------------------------------ 主流程
    def run(self, srcs):
        if not srcs:
            raise SystemExit("[输入] 没有找到任何图片")

        keys = build_keys(srcs)
        self.stats.update(total=len(srcs), done=0, skipped=0, failed=0)

        todo = []
        for i, key in enumerate(keys):
            if key in self.done:
                self.stats["skipped"] += 1
                continue
            if self.retry_failed and key not in self.failed:
                self.stats["skipped"] += 1      # 只重跑上次失败的
                continue
            todo.append(i)

        print("[计划] 共 %d 张 | 跳过 %d 张（已完成）| 本轮要跑 %d 张 | bs=%d"
              % (len(srcs), self.stats["skipped"], len(todo), self.batch_size))
        if not todo:
            print("[计划] 没有需要跑的图，直接汇总。")
            return self._finish(None)

        t0 = time.time()
        self._pool = ThreadPoolExecutor(max_workers=self.prefetch)
        try:
            for b0 in range(0, len(todo), self.batch_size):
                chunk = todo[b0:b0 + self.batch_size]
                self._run_one_batch(chunk, srcs, keys, b0 // self.batch_size + 1,
                                    t0, len(todo))
        finally:
            self._pool.shutdown(wait=True)
            self._pool = None

        return self._finish(t0)

    def _run_one_batch(self, chunk, srcs, keys, batch_id, t0, n_todo):
        t_batch = time.time()
        paths = [srcs[i] for i in chunk]

        # 读图 + 预处理在线程池里并行（PIL 解码会释放 GIL），GPU 前向仍然串行
        preps = list(self._pool.map(self._safe_prep, paths))

        kept = []
        for i, prep in zip(chunk, preps):
            if isinstance(prep, BaseException):
                self._record_failed(keys[i], srcs[i], prep)
            else:
                kept.append((i, prep))

        if kept:
            xs = np.stack([prep["x"] for _, prep in kept], axis=0)
            preds = self._infer_with_fallback(xs)
            per_img = (time.time() - t_batch) / max(1, len(chunk))
            for (i, prep), pred in zip(kept, preds):
                if isinstance(pred, BaseException):
                    self._record_failed(keys[i], srcs[i], pred)
                else:
                    self._commit(keys[i], srcs[i], prep, pred, batch_id, per_img)

        self._progress(t0, n_todo)

    # ------------------------------------------------------------------ 步骤
    def _safe_prep(self, path):
        try:
            return self.adapter.prep(path)
        except Exception as exc:                # 坏图 / 位深不支持 / 文件被占用
            return exc

    def _infer_with_fallback(self, xs):
        """整批失败就对半拆开重试，拆到 1 张还失败才记入失败清单。

        这样单张坏图（或偶发显存不足）不会连带毁掉整批结果。
        """
        try:
            preds = self.adapter.infer(xs)
            if len(preds) != len(xs):
                raise RuntimeError("infer 返回 %d 条，输入 %d 条" % (len(preds), len(xs)))
            return list(preds)
        except Exception as exc:
            if len(xs) == 1:
                return [exc]
            # 显存不足要先把缓存块还给驱动，否则对半拆完照样 OOM
            oom = ("OutOfMemory" in type(exc).__name__
                   or "out of memory" in str(exc).lower())
            if oom and hasattr(self.adapter, "on_oom"):
                self.adapter.on_oom()
            mid = len(xs) // 2
            if self.verbose:
                print("[警告] %d 张一批推理失败（%s: %s），拆成两半重试"
                      % (len(xs), type(exc).__name__, exc))
                if oom:
                    print("[提示] 这是显存不足：512 输入下 U-Net 建议 --batch-size 4~8、"
                          "SegFormer 建议 8~16；设太大会让显卡把显存溢出到内存，"
                          "系统内存冲到 90% 以上且吞吐掉一个数量级")
            return (self._infer_with_fallback(xs[:mid])
                    + self._infer_with_fallback(xs[mid:]))

    def _commit(self, key, src, prep, mask_small, batch_id, per_img):
        meta = prep["meta"]
        mask = np.asarray(self.adapter.post(mask_small, meta), dtype=np.uint8)

        row = dict(key=key, src=src, H=mask.shape[0], W=mask.shape[1],
                   fg_ratio="%.6f" % float((mask > 0).mean()),
                   ms="%.1f" % (per_img * 1000.0),
                   batch_id=batch_id, ts=now_str(), mask="", overlay="")

        if self.save_mask:
            arr = mask if self.mask_mode == "index" else self.adapter.colorize(mask)
            rel = os.path.join("masks", key + "_mask" + self.mask_ext)
            save_image_atomic(os.path.join(self.out, rel), arr)
            row["mask"] = rel

        if self.save_overlay:
            rel = os.path.join("overlays", key + "_overlay.tif")
            save_image_atomic(os.path.join(self.out, rel),
                              self.adapter.overlay(mask, src))
            row["overlay"] = rel

        # ---- 提交点：产物都落盘之后才写这一行 ----
        self._append_row(self.manifest_path, MANIFEST_FIELDS, row)
        self.done[key] = row
        self.failed.pop(key, None)
        self.stats["done"] += 1

    def _record_failed(self, key, src, err):
        prev = self.failed.get(key) or {}
        try:
            attempts = int(prev.get("attempts", 0)) + 1
        except (TypeError, ValueError):
            attempts = 1
        row = dict(key=key, src=src,
                   error="%s: %s" % (type(err).__name__, err),
                   attempts=attempts, ts=now_str())
        self.failed[key] = row
        self._append_row(self.failed_path, FAILED_FIELDS, row)
        self.stats["failed"] += 1

    def _progress(self, t0, n_todo):
        if not self.verbose:
            return
        processed = self.stats["done"] + self.stats["failed"]
        if processed % self.log_every and processed != n_todo:
            return
        el = time.time() - t0
        speed = processed / el if el > 0 else 0.0
        eta = (n_todo - processed) / speed if speed > 0 else 0.0
        print("[%5d/%-5d] 成功 %d 失败 %d | %.1f img/s | 已用 %s | 剩余 %s"
              % (processed, n_todo, self.stats["done"], self.stats["failed"],
                 speed, fmt_dur(el), fmt_dur(eta)))

    # ------------------------------------------------------------------ 落盘读写
    @staticmethod
    def _append_row(path, fields, row):
        new_file = (not os.path.exists(path)) or os.path.getsize(path) == 0
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                f.write("\ufeff")               # BOM，Excel 打开不乱码
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})

    def _read_rows(self, path):
        rows = []
        if not os.path.exists(path):
            return rows
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows

    def _read_done(self):
        done = {}
        for row in self._read_rows(self.manifest_path):
            key = (row.get("key") or "").strip()
            if key and self._artifacts_ok(row):
                done[key] = row
        return done

    def _artifacts_ok(self, row):
        """manifest 有行还不够：产物要真在、且不是空文件。"""
        need = []
        if self.save_mask:
            need.append(row.get("mask"))
        if self.save_overlay:
            need.append(row.get("overlay"))
        for rel in need:
            if not rel:
                return False
            path = os.path.join(self.out, rel)
            if not os.path.isfile(path) or os.path.getsize(path) < 128:
                return False
        return True

    def _read_failed(self):
        failed = {}
        for row in self._read_rows(self.failed_path):
            key = (row.get("key") or "").strip()
            if key:
                failed[key] = row
        return failed

    def _compact_failed(self):
        """failed.csv 是追加写的，这里压一遍：每张图只留最后一次，已成功的剔掉。"""
        rows = [r for k, r in self.failed.items() if k not in self.done]
        if not rows:
            if os.path.exists(self.failed_path):
                save_text_atomic(self.failed_path, "")
            return

        def writer(tmp):
            with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FAILED_FIELDS)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in FAILED_FIELDS})

        _atomic_write(self.failed_path, writer)

    # ------------------------------------------------------------------ 汇总
    def _finish(self, t0):
        self._compact_failed()
        el = (time.time() - t0) if t0 else 0.0
        done = self.stats["done"]
        failed = self.stats["failed"]
        skipped = self.stats["skipped"]

        summary = dict(
            finished=now_str(),
            elapsed_s=round(el, 1),
            total=self.stats["total"],
            done=done,
            failed=failed,
            skipped_already_done=skipped,
            img_per_s=round(done / el, 3) if el > 0 else 0.0,
            ms_per_img=round(el * 1000.0 / done, 1) if done else 0.0,
            out=self.out,
        )
        save_json_atomic(os.path.join(self.out, "summary.json"), summary)

        print("-" * 74)
        print("[汇总] 成功 %d | 失败 %d | 跳过(已完成) %d | 用时 %s"
              % (done, failed, skipped, fmt_dur(el)))
        if done:
            print("[汇总] 吞吐 %.2f img/s | 平均 %.1f ms/张"
                  % (summary["img_per_s"], summary["ms_per_img"]))
        print("[汇总] 清单 %s" % self.manifest_path)
        if failed:
            print("[汇总] 失败清单 %s（用 --retry-failed 只重跑这些）" % self.failed_path)
            for key in list(self.failed)[:5]:
                print("        %s : %s" % (key, self.failed[key].get("error", "")))
        print("[汇总] 输出目录 %s" % self.out)
        return summary
