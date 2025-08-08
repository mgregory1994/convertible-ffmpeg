from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock, Thread
from typing import Iterable
from queue import Queue

from ffmpeg import Progress, MediaTask, FolderTask, FFmpegHelper


class TaskQueue:
    def __new__(cls, max_workers: int = 1) -> TaskQueue:
        if not hasattr(cls, "instance"):
            cls.instance = super(TaskQueue, cls).__new__(cls)
            cls._max_workers: int = max_workers
            cls._task_queue: Queue[MediaTask | FolderTask | bool] = Queue()
            cls._running_tasks: list[MediaTask | FolderTask] = []
            cls._task_lock: Lock = Lock()

            Thread(
                target=cls._start_threadpool, args=(cls.instance,), daemon=True
            ).start()

        return cls.instance

    def _start_threadpool(self):
        with ThreadPoolExecutor(self._max_workers) as executor:
            for _ in range(self._max_workers):
                executor.submit(self._run_loop_instance)

    def _get_task(self) -> Iterable[MediaTask | FolderTask | bool]:
        yield self.task_queue.get()

    def _run_loop_instance(self):
        for task in self._get_task():
            if isinstance(task, bool):
                break

            self._add_running_task(task)

            if isinstance(task, FolderTask):
                self._execute_folder_task_ffmpeg(task)
            else:
                self._execute_task_ffmpeg(task)

            self._remove_running_task(task)
            self.task_queue.task_done()

    @property
    def task_queue(self) -> Queue[MediaTask | FolderTask | bool]:
        return self._task_queue

    @property
    def running_tasks(self) -> list[MediaTask | FolderTask]:
        with self._task_lock:
            return self._running_tasks.copy()

    def add_task(self, task: MediaTask | FolderTask) -> None:
        self.task_queue.put(task)

    def _add_running_task(self, task: MediaTask | FolderTask) -> None:
        with self._task_lock:
            self._running_tasks.append(task)

    def _remove_running_task(self, task: MediaTask | FolderTask) -> None:
        try:
            with self._task_lock:
                self._running_tasks.remove(task)
        except ValueError:
            pass

    def pause_running_tasks(self) -> None:
        with self._task_lock:
            for task in self._running_tasks:
                task.is_paused = True

    def resume_running_tasks(self) -> None:
        with self._task_lock:
            for task in self._running_tasks:
                task.is_paused = False

    def stop_running_tasks(self) -> None:
        with self._task_lock:
            for task in self._running_tasks:
                task.is_stopped = True

    def join(self) -> None:
        self.task_queue.join()

    def kill_task_queue(self) -> None:
        self._empty_task_queue()
        self.resume_running_tasks()
        self.stop_running_tasks()
        self._stop_running_task_threads()

    def _empty_task_queue(self) -> None:
        while not self.task_queue.empty():
            self.task_queue.get()
            self.task_queue.task_done()

    def _stop_running_task_threads(self) -> None:
        for _ in range(self._max_workers):
            self.task_queue.put(False)

    def _execute_task_ffmpeg(self, task: MediaTask) -> None:
        FFmpegHelper.initialize_ffmpeg(task)

        @task.ffmpeg.on("progress")
        def _(progress: Progress):
            task.progress = progress

        task.execute_ffmpeg()

    def _execute_folder_task_ffmpeg(self, folder_task: FolderTask) -> None:
        for task in folder_task.media_folder.media_tasks:
            self._execute_task_ffmpeg(task)


class HwaQueue(TaskQueue): ...


class WatchFolderQueue(TaskQueue):
    def _execute_folder_task_ffmpeg(self, folder_task: FolderTask):
        Thread(target=WatchFolderQueue._execute_watch_folder_task_ffmpeg, args=(folder_task,)).start()

    def _execute_watch_folder_task_ffmpeg(self, folder_task: FolderTask):
        while not folder_task.is_stopped:
            task = folder_task.task_queue.get()
            self._add_running_task(task)
            self._execute_task_ffmpeg(task)

            if task.is_error and not folder_task.is_error:
                folder_task.is_error = True

            self._remove_running_task(task)

        folder_task.media_folder.observer.stop()
