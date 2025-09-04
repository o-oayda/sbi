import math
from collections import deque
from typing import Optional
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn


class MultiRoundInfererUI:
    def __init__(
            self,
            tasks: list[str],
            steps_with_progress: Optional[set] = None, 
            title: str = "Multi-round Inferer", 
            log_lines: int = 8
    ) -> None:
        self.tasks = list(tasks)
        self.title = title
        self.finished = 0
        self.current = 0
        self.subtitle = "starting"
        self.steps_with_progress = set(steps_with_progress or [])
        self._progress = None
        self._task_id = None
        self._logs = deque(maxlen=log_lines)

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

        if idx in self.steps_with_progress:
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
            self._progress.stop()
            self._progress.tasks.clear()
            self._task_id = self._progress.add_task(
                self.tasks[idx],
                total=(total or 100),
                start=False
            )
        else:
            self._task_id = None

    def advance_progress(self, advance: int = 1, total: Optional[int] = None) -> None:
        if self._progress is not None and self._task_id is not None:

            if total is not None and math.isfinite(total):
                self._progress.update(self._task_id, total=total)

            if not self._progress.tasks[self._task_id].started:
                self._progress.start_task(self._task_id)
            self._progress.update(self._task_id, advance=advance)

    def set_subtitle(self, text: str) -> None:
        self.subtitle = text

    def log(self, message: str, style: str = "") -> None:
        # style examples: "dim", "yellow", "red", "green", "italic cyan"
        t = (
                Text.from_markup(message) if '[' in message and ']' in message
                else Text(message)
        )
        if style:
            t.stylize(style)
        self._logs.append(t)

    def finish_step(self, subtitle: Optional[str] = None) -> None:
        self.finished = max(self.finished, self.current + 1)
        if subtitle is not None:
            self.subtitle = subtitle
        if self._progress is not None and self._task_id is not None:
            remaining = (
                self._progress.tasks[self._task_id].total
              - self._progress.tasks[self._task_id].completed
            )
            if remaining > 0:
                self._progress.update(self._task_id, advance=remaining)

    def is_done(self):
        return self.finished >= len(self.tasks)

    # ---------- rendering ----------
    def _render_checklist(self) -> Table:
        table = Table.grid(padding=(0, 1))
        for i, label in enumerate(self.tasks):
            if i < self.finished:
                mark = "[green]✓[/]"
                style = "green"
            elif i == self.current:
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

    def render(self) -> Panel:
        body = [self._render_checklist()]
        if self._task_id is not None and self.current in self.steps_with_progress:
            body.append(self._progress)
        body.append(self._render_logs())
        return Panel(
            Group(*body),
            title=self.title,
            subtitle=self.subtitle,
            border_style="white",
        )
