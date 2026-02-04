# -*- coding: utf-8 -*-

import logging, json, time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, Iterable, Sequence, Union

import numpy as np
import os


# ---------- config ----------
@dataclass
class LogConfig:
    enabled: bool = True

    # printing (independent of enabled)
    print_to_console: bool = False
    level: int = logging.INFO

    # frequency controls (you already use these in the algorithm)
    log_every: int = 200
    outer_log_every: int = 1

    # storage controls
    store_csv: bool = False
    store_jsonl: bool = True

    # paths: if None + store_* True => auto under root_dir
    csv_path: Optional[str] = None
    jsonl_path: Optional[str] = None

    # run folder handling
    root_dir: str = "logs"
    run_dir: Optional[str] = None     # if set, use it as run folder
    run_name: Optional[str] = None    # included in folder name + meta
    meta: Dict[str, Any] = field(default_factory=dict)

    # callback
    callback: Optional[Callable[[Dict[str, Any]], None]] = None

    # memory buffer for later visualization
    keep_in_memory: bool = True

    # avoid dumping huge configs (like I matrices)
    include_cfg: bool = True



def build_default_logger(
    name: str = "score_structure_learning",
    level: int = logging.INFO,
    stream: bool = True,
    logfile: Optional[str] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if getattr(logger, "_configured", False):
        return logger

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    if stream:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger._configured = True
    return logger

class StructuredLogger:
    def __init__(self, logger: logging.Logger, cfg: LogConfig):
        self.logger = logger
        self.cfg = cfg

        # in-memory storage for later plotting
        self._rows = [] if cfg.keep_in_memory else None

        # decide run directory if we need any file output
        needs_files = (cfg.store_csv or cfg.store_jsonl) and cfg.enabled
        self.run_dir = None

        if needs_files:
            self.run_dir = self._resolve_run_dir()
            os.makedirs(self.run_dir, exist_ok=True)

            # write meta.json once
            meta_path = os.path.join(self.run_dir, "meta.json")
            meta = {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "run_name": cfg.run_name,
                **(cfg.meta or {}),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        # open sinks
        self.jsonl_path = None
        self.csv_path = None

        self._jsonl_f = None
        if cfg.enabled and cfg.store_jsonl:
            self.jsonl_path = cfg.jsonl_path or (os.path.join(self.run_dir, "metrics.jsonl") if self.run_dir else None)
            if self.jsonl_path:
                self._jsonl_f = open(self.jsonl_path, "a", encoding="utf-8")

        self._csv_f = None
        self._csv_header_written = False
        if cfg.enabled and cfg.store_csv:
            self.csv_path = cfg.csv_path or (os.path.join(self.run_dir, "metrics.csv") if self.run_dir else None)
            if self.csv_path:
                import csv
                self._csv_mod = csv
                self._csv_f = open(self.csv_path, "a", newline="", encoding="utf-8")

    def _resolve_run_dir(self) -> str:
        if self.cfg.run_dir is not None:
            return self.cfg.run_dir

        ts = time.strftime("%Y%m%d-%H%M%S")
        name = (self.cfg.run_name or "run").replace(" ", "_")
        # add a short monotonic suffix to avoid collisions
        suffix = str(int(time.time() * 1000) % 100000)
        folder = f"{ts}_{name}_{suffix}"
        return os.path.join(self.cfg.root_dir, folder)

    def close(self):
        if self._jsonl_f:
            self._jsonl_f.close()
        if self._csv_f:
            self._csv_f.close()

    def emit(self, event: str, metrics: Dict[str, Any]):
        if not self.cfg.enabled:
            return

        row = {"event": event, **metrics}

        # store in memory
        if self._rows is not None:
            self._rows.append(row)

        # print only if requested
        if self.cfg.print_to_console:
            self.logger.log(self.cfg.level, f"{event} | " + self._fmt(metrics))

        # write jsonl
        if self._jsonl_f:
            self._jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._jsonl_f.flush()

        # write csv
        if self._csv_f:
            w = self._csv_mod.DictWriter(self._csv_f, fieldnames=list(row.keys()))
            if not self._csv_header_written:
                w.writeheader()
                self._csv_header_written = True
            w.writerow(row)
            self._csv_f.flush()

        if self.cfg.callback:
            try:
                self.cfg.callback(row)
            except Exception:
                self.logger.exception("logging callback failed")

    @staticmethod
    def _fmt(d: Dict[str, Any]) -> str:
        parts = []
        for k, v in d.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4e}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)
    
    def load(self, *, source: Optional[str] = None,
             event: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """
        Load rows into column arrays.
        Priority:
          1) memory buffer (if exists and source is None)
          2) source file
          3) jsonl_path
          4) csv_path
        """
        rows = None
    
        if source is None and self._rows is not None and len(self._rows) > 0:
            rows = list(self._rows)
        else:
            path = source or self.jsonl_path or self.csv_path
            if path is None:
                raise ValueError("No logs in memory and no file path available.")
            if path.endswith(".jsonl"):
                rows = self._load_jsonl(path)
            elif path.endswith(".csv"):
                rows = self._load_csv(path)
            else:
                # attempt jsonl then csv
                try:
                    rows = self._load_jsonl(path)
                except Exception:
                    rows = self._load_csv(path)
    
        # filter by event(s)
        if event is not None:
            if isinstance(event, str):
                events = {event}
            else:
                events = set(event)
            rows = [r for r in rows if r.get("event") in events]
    
        if not rows:
            raise ValueError("No rows found (after filtering).")
    
        keys = set()
        for r in rows:
            keys.update(r.keys())
    
        cols = {k: [] for k in keys}
        for r in rows:
            for k in keys:
                cols[k].append(r.get(k, None))
    
        return {k: np.array(v, dtype=object) for k, v in cols.items()}
    
    def visualize(
        self,
        *,
        event="minimize.checkpoint",
        source: Optional[str] = None,
        x: str = "iter",
        group: Optional[str] = "stage",
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        ncols: int = 2,
        smooth: int = 1,
        figsize: Optional[tuple] = None,
        sharex: bool = True,
        show: bool = True,
        save_path: Optional[str] = None,
        max_plots: Optional[int] = None,
    ):
        """
        General dashboard plot:
        - auto-detect numeric metrics (unless include is given)
        - plot all in one big figure with subplots
        """
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Dashboard plotting requires matplotlib. Install with:\n"
                "  conda install matplotlib\n"
                "or run without plotting."
            )
    
        import numpy as np
        import math
        import os
    
        data = self.load(source=source, event=event)
    
        def is_numericish(v) -> bool:
            if v is None:
                return True
            if isinstance(v, (int, float, np.number)):
                return True
            if isinstance(v, (dict, list, tuple, set)):
                return False
            try:
                float(v)
                return True
            except Exception:
                return False
    
        def col_numeric_fraction(arr: np.ndarray) -> float:
            ok = 0
            tot = 0
            for v in arr:
                if v is None:
                    tot += 1
                    continue
                tot += 1
                ok += int(is_numericish(v))
            return ok / max(tot, 1)
    
        def to_float(arr: np.ndarray) -> np.ndarray:
            out = np.empty(len(arr), dtype=float)
            for i, v in enumerate(arr):
                if v is None:
                    out[i] = np.nan
                elif isinstance(v, (int, float, np.number)):
                    out[i] = float(v)
                else:
                    try:
                        out[i] = float(v)
                    except Exception:
                        out[i] = np.nan
            return out
    
        def moving_avg(y: np.ndarray, w: int) -> np.ndarray:
            if w <= 1:
                return y
            out = np.full_like(y, np.nan, dtype=float)
            for i in range(len(y)):
                out[i] = np.nanmean(y[max(0, i - w + 1): i + 1])
            return out
    
        # x axis
        if x not in data:
            raise ValueError(f"x='{x}' not found. Available keys: {sorted(data.keys())}")
        xvals = to_float(data[x])
    
        # group labels
        if group is not None and group in data:
            glabels = np.array([str(v) if v is not None else "None" for v in data[group]], dtype=object)
            groups = sorted(set(glabels.tolist()))
        else:
            group = None
            glabels = np.array(["all"] * len(xvals), dtype=object)
            groups = ["all"]
    
        include_set = set(include) if include is not None else None
        exclude_set = set(exclude) if exclude is not None else set()
    
        # choose y-metrics
        metrics = []
        for k, arr in data.items():
            if k in (x, group, "event"):
                continue
            if include_set is not None and k not in include_set:
                continue
            if k in exclude_set:
                continue
            if k.endswith("_cfg") or k.endswith("_name"):
                continue
            if col_numeric_fraction(arr) >= 0.6:
                metrics.append(k)
    
        metrics.sort()
        if max_plots is not None:
            metrics = metrics[: int(max_plots)]
    
        if not metrics:
            raise ValueError("No numeric metrics found to plot (after filters).")
    
        # dashboard grid
        n = len(metrics)
        ncols = max(1, int(ncols))
        nrows = int(math.ceil(n / ncols))
        if figsize is None:
            figsize = (6.5 * ncols, 3.2 * nrows)
    
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, squeeze=False)
    
        for idx, m in enumerate(metrics):
            r, c = divmod(idx, ncols)
            ax = axes[r][c]
    
            y = moving_avg(to_float(data[m]), smooth)
    
            for g in groups:
                mask = (glabels == g)
                xx, yy = xvals[mask], y[mask]
                order = np.argsort(xx)
                ax.plot(xx[order], yy[order], label=(f"{group}={g}" if group else None))
    
            ax.set_title(m)
            ax.set_ylabel(m)
            ax.set_yscale("linear")   # "linear" | "log" | "symlog"
            
            if r == nrows - 1:
                ax.set_xlabel(x)
    
        # hide unused
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r][c].axis("off")
    
        # one legend
        if group:
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right")
    
        fig.tight_layout()
    
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
        if show:
            plt.show()
        else:
            plt.close(fig)


    
    @staticmethod
    def _load_jsonl(path: str):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    
    @staticmethod
    def _load_csv(path: str):
        import csv
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(dict(r))
        return rows




# ---------- metric schema helpers ----------
@dataclass(frozen=True)
class RegularizerInfo:
    name: str                      # e.g. "dagma_logdet", "notears_exp", "spectral_radius"
    cfg: Dict[str, Any] = field(default_factory=dict)

def w_stats(W: np.ndarray) -> Dict[str, float]:
    W = np.asarray(W)
    absW = np.abs(W)
    nz = absW[np.nonzero(absW)]
    return dict(
        w_norm=float(np.linalg.norm(W)),
        w_abs_sum=float(absW.sum()),
        max_abs_w=float(absW.max()) if absW.size else 0.0,
        min_abs_w_nonzero=float(nz.min()) if nz.size else 0.0,
    )

def build_common_metrics(
    *,
    iter: int,
    stage: int,
    elapsed_sec: float,
    W: np.ndarray,
    obj_total: Optional[float] = None,
    score_datafit: Optional[float] = None,
    dag_reg_value: Optional[float] = None,
    dag_reg: Optional[RegularizerInfo] = None,
    trek_reg_value: Optional[float] = None,
    trek_reg: Optional[RegularizerInfo] = None,
    extras: Optional[Dict[str, Any]] = None,
    include_cfg: bool = True,
) -> Dict[str, Any]:
    m: Dict[str, Any] = dict(iter=int(iter), stage=int(stage), elapsed_sec=float(elapsed_sec))
    m.update(w_stats(W))

    if obj_total is not None:
        m["obj_total"] = float(obj_total)
    if score_datafit is not None:
        m["score_datafit"] = float(score_datafit)

    if dag_reg is not None:
        m["reg_dag_name"] = dag_reg.name
        if include_cfg:
            m["reg_dag_cfg"] = dag_reg.cfg
    if dag_reg_value is not None:
        m["reg_dag_value"] = float(dag_reg_value)

    if trek_reg is not None:
        m["reg_trek_name"] = trek_reg.name
        if include_cfg:
            m["reg_trek_cfg"] = trek_reg.cfg
    if trek_reg_value is not None:
        m["reg_trek_value"] = float(trek_reg_value)

    if extras:
        m.update(extras)

    return m
