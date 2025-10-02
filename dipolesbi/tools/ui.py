from contextlib import contextmanager
from collections import deque
from threading import Lock
from typing import Optional, Sequence, Mapping
from rich.align import Align
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.columns import Columns


class MultiRoundInfererUI:
    def __init__(
            self,
            tasks: list[str],
            title: str = "Multi-round Inferer", 
            log_lines: int = 8
    ) -> None:
        self.tasks = list(tasks)
        self.title = title
        self.finished = 0
        self.current = None
        self.subtitle = ""
        self._progress = None
        self._task_id = None
        self._logs = deque(maxlen=log_lines)
        self._live = None
        self._lock = Lock()
        self._global_prog = None
        self._global_task = None
        self.global_desc = "Simulations"
        self._stats_columns: list[str] = ["Round", "Value"]
        self._stats_rows: list[list[str]] = []

        self.round_idx = None
        self.round_total = None

    @contextmanager
    def session(self, refresh_per_second: int = 20, transient: bool = False):
        from rich.live import Live
        with Live(self.render(), refresh_per_second=refresh_per_second, transient=transient) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None

    def _refresh(self):
        # safe to call from anywhere in same thread; guard just in case
        if self._live is not None:
            self._live.update(self.render())

    # ---------- public API ----------
    def start_step(
            self, 
            idx: int, 
            subtitle: Optional[str] = None,
            total: Optional[int] = None
    ) -> None:
        self.current = idx
        if subtitle is not None:
            self.subtitle = subtitle
        self._task_id = None
        self._refresh()

    def set_subtitle(self, text: str) -> None:
        self.subtitle = text
        self._refresh()

    def log(self, message: str, style: str = "") -> None:
        # style examples: "dim", "yellow", "red", "green", "italic cyan"
        t = Text.from_markup(message) if '[' in message and ']' in message else Text(message)
        if style:
            t.stylize(style)
        self._logs.append(t)
        self._refresh()

    def finish_step(self, subtitle: Optional[str] = None) -> None:
        self.finished = max(self.finished, self.current + 1)
        if subtitle is not None:
            self.subtitle = subtitle
        if self._progress is not None and self._task_id is not None:
            remaining = (self._progress.tasks[self._task_id].total
                         - self._progress.tasks[self._task_id].completed)
            if remaining > 0:
                self._progress.update(self._task_id, advance=remaining)
        self._refresh()

    def is_done(self):
        return self.finished >= len(self.tasks)

    def set_round(self, idx=None, total=None):
        if idx is not None:
            self.round_idx = idx
        if total is not None:
            self.round_total = total
        self._refresh()

    def increment_round(self, step=1):
        self.round_idx = (
            0 if self.round_idx is None
            else self.round_idx
        ) + step
        self._refresh()

    def reset(self, subtitle: str = "ready"):
        self.finished = 0
        self.current = None
        self.subtitle = subtitle
        self._progress = None
        self._task_id = None
        self._refresh()

    def begin_progress(self, total: Optional[int] = 100, description=None):
        """Create (or re-create) a progress bar for the *current* step."""
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
        if self.current is None:
            raise RuntimeError("No active step. Call start_step(...) first.")

        if self._progress is None:
            self._progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
                expand=True,
            )
        else:
            self._progress.stop()
            self._progress.tasks.clear()

        desc = description or self.tasks[self.current]
        self._task_id = self._progress.add_task(desc, total=total, start=False)
        self._refresh()

    def update_progress(self, advance=None, completed=None, total=None):
        """Advance or set progress for the current step's bar."""
        if self._progress is None or self._task_id is None:
            return
        kw = {}
        if total is not None: kw["total"] = total
        if completed is not None: kw["completed"] = completed
        if not self._progress.tasks[self._task_id].started:
            self._progress.start_task(self._task_id)
        if advance is not None:
            self._progress.update(self._task_id, advance=advance, **kw)
        elif kw:
            self._progress.update(self._task_id, **kw)
        self._refresh()

    def end_progress(self, complete=True):
        """Finish and hide the bar for the current step."""
        if self._progress is None or self._task_id is None:
            return
        if complete:
            t = self._progress.tasks[self._task_id]
            remaining = (t.total - t.completed) if (t.total is not None) else 0
            if remaining > 0:
                    self._progress.update(self._task_id, advance=remaining)
        # Keep the Progress instance around so render() can still include it this frame
        # but clear the task so it disappears on next render.
        self._progress.tasks.clear()
        self._task_id = None
        self._refresh()

# ----------- simulation progress bar -------------
    def begin_global_progress(self, total: int | None, description: str | None = None):
        """Create/replace a global progress bar (indeterminate if total=None)."""
        if self._global_prog is None:
            self._global_prog = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}" + ("/{task.total}" if total is not None else "")),
                TextColumn("{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
                expand=True,
            )
        else:
            self._global_prog.stop()
            self._global_prog.tasks.clear()
        desc = description or self.global_desc
        # NOTE: total=None → indeterminate bar
        self._global_task = self._global_prog.add_task(desc, total=total, start=True)
        self._refresh()

    def set_global_total(self, total: int | None):
        """Switch between determinate and indeterminate by setting total."""
        if self._global_prog is None or self._global_task is None:
            return
        self._global_prog.update(self._global_task, total=total)
        self._refresh()

    def set_global_description(self, description: str):
        self.global_desc = description
        if self._global_prog is not None and self._global_task is not None:
            self._global_prog.update(self._global_task, description=description)
            self._refresh()

    def advance_global(self, n: int = 1):
        if self._global_prog is None or self._global_task is None:
            return
        self._global_prog.update(self._global_task, advance=n)
        self._refresh()

    def set_global_completed(self, completed: int):
        if self._global_prog is None or self._global_task is None:
            return
        self._global_prog.update(self._global_task, completed=completed)
        self._refresh()

    def end_global_progress(self, complete: bool = True):
        if self._global_prog is None or self._global_task is None:
            return
        if complete:
            t = self._global_prog.tasks[self._global_task]
            if t.total is not None:
                remaining = t.total - t.completed
                if remaining > 0:
                    self._global_prog.update(self._global_task, advance=remaining)
        # remove the task so the bar disappears next render
        self._global_prog.tasks.clear()
        self._global_task = None
        self._refresh()


# ------------------- evidence panel --------------------
    def set_stats_columns(self, columns: Sequence[str]) -> None:
        """Define the table schema (call once, e.g., at startup)."""
        self._stats_columns = list(columns)
        self._refresh()

    def add_stats_row(self, row: Sequence[object] | Mapping[str, object]) -> None:
        """
        Append one row. Accepts a sequence (must match column order) or a mapping
        (keys matched to columns; missing keys become '').
        """
        if isinstance(row, Mapping):
            ordered = [row.get(col, "") for col in self._stats_columns]
        else:
            ordered = list(row)
            # pad/trim to fit schema
            if len(ordered) < len(self._stats_columns):
                ordered += [""] * (len(self._stats_columns) - len(ordered))
            elif len(ordered) > len(self._stats_columns):
                ordered = ordered[:len(self._stats_columns)]
        # stringify with light formatting
        self._stats_rows.append([self._fmt_cell(v) for v in ordered])
        self._refresh()

    def update_last_stats_row(self, updates: Mapping[str, object]) -> None:
        """Update specific columns in the most recently added row."""
        if not self._stats_rows:
            return
        last = self._stats_rows[-1]
        for k, v in updates.items():
            try:
                idx = self._stats_columns.index(k)
            except ValueError:
                continue
            last[idx] = self._fmt_cell(v)
        self._refresh()

    def clear_stats(self) -> None:
        """Manually clear the accumulated metrics table."""
        self._stats_rows.clear()
        self._refresh()

    def _fmt_cell(self, v: object) -> str:
        # Nice defaults for numbers; tweak if you like.
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    # ---------- rendering ----------
    def _render_checklist(self) -> Table:
        table = Table.grid(padding=(0, 1))
        for i, label in enumerate(self.tasks):
            if i < self.finished:
                mark = "[green]✓[/]"
                style = "green"
            elif self.current is not None and i == self.current:
                mark = "[cyan]…[/]"
                style = "cyan"
            else:
                mark = " "
                style = "dim"
            table.add_row(f"[{mark}] [bold {style}]{label}[/]")
        return table

    def _render_logs(self) -> Panel:
        if not self._logs:
            return Panel("No logs yet.", title="Logs", border_style="dim")
        # Newest at bottom; show as a simple grid
        log_table = Table.grid(padding=(0, 1))
        for line in self._logs:
            log_table.add_row(line)
        return Panel(log_table, title="Logs", border_style="grey50")

    def _render_header(self):
        t = Table.grid(padding=(0,1))
        cur_num = self.round_idx + 1 if self.round_idx is not None else '-'
        n_rounds = f"/{self.round_total}" if (self.round_total is not None) else ""
        left = f"[bold]Round:[/] {cur_num}{n_rounds}"
        t.add_row(left)
        return Align.left(t)

    def _render_stats(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        # Header row:
        header = Table.grid()
        header.add_row("[bold]Round Metrics[/bold]")
        # Data table:
        data = Table(*[c for c in self._stats_columns], expand=True)
        for row in self._stats_rows:
            data.add_row(*row)
        return Panel(Group(header, data), border_style="grey50")

    def render(self) -> Panel:
        left_body = [self._render_checklist()]
        left_panel = Panel(Group(*left_body), border_style='white')

        right_panel = None
        if self._stats_rows:
            right_panel = self._render_stats()

        row = Columns([left_panel, right_panel] if right_panel else [left_panel], expand=True)

        body = [self._render_header()]
        if self._global_task is not None:
            body.append(self._global_prog)
        if self._task_id is not None:
            body.append(self._progress)
        body.append(self._render_logs())
        body.append(row)

        return Panel(
            Group(*body),
            title=self.title,
            subtitle=self.subtitle,
            border_style="white",
        )


class NullMultiRoundInfererUI:
    def __init__(self, tasks: Optional[list[str]] = None, *_, **__) -> None:
        self.tasks = tasks or []

    @contextmanager
    def session(self, *_, **__):
        yield self

    # no-op implementations of the API used by the real UI
    def start_step(self, *_, **__):
        return None

    def finish_step(self, *_, **__):
        return None

    def log(self, *_, **__):
        return None

    def set_round(self, *_, **__):
        return None

    def increment_round(self, *_, **__):
        return None

    def reset(self, *_, **__):
        return None

    def set_subtitle(self, *_, **__):
        return None

    def set_stats_columns(self, *_, **__):
        return None

    def add_stats_row(self, *_, **__):
        return None

    def update_last_stats_row(self, *_, **__):
        return None

    def begin_progress(self, *_, **__):
        return None

    def update_progress(self, *_, **__):
        return None

    def end_progress(self, *_, **__):
        return None

    def begin_global_progress(self, *_, **__):
        return None

    def set_global_total(self, *_, **__):
        return None

    def set_global_description(self, *_, **__):
        return None

    def advance_global(self, *_, **__):
        return None

    def set_global_completed(self, *_, **__):
        return None

    def end_global_progress(self, *_, **__):
        return None
