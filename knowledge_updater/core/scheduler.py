import time
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, List
import threading
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor

from .config import get_config
from .logging import get_logger, log_performance

logger = get_logger(__name__)


class KnowledgeUpdateScheduler:
    def __init__(self):
        self.config = get_config()
        self.scheduler = None
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._running = False
        self._data_manager = None
        self._vector_manager = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                logger.warning("Scheduler is already running")
                return

            try:
                scheduler_config = self.config.get('scheduler', {})
                max_workers = min(scheduler_config.get('max_concurrent_jobs', 3), 10)

                self.scheduler = BackgroundScheduler(
                    jobstores={
                        'default': MemoryJobStore()
                    },
                    executors={
                        'default': ThreadPoolExecutor(max_workers=max_workers)
                    },
                    job_defaults={
                        'coalesce': True,
                        'max_instances': 3,
                        'misfire_grace_time': 30
                    }
                )

                self._schedule_periodic_updates()
                self.scheduler.start()
                self._running = True

                logger.info(
                    "Knowledge update scheduler started",
                    max_workers=max_workers,
                    update_interval=f"{scheduler_config.get('update_interval_hours', 24)}h"
                )

            except Exception as e:
                logger.error("Failed to start scheduler", error=str(e))
                raise

    def stop(self, wait: bool = True) -> None:
        with self._lock:
            if not self._running or not self.scheduler:
                return

            logger.info("Stopping knowledge update scheduler")
            self.scheduler.shutdown(wait=wait)
            self._running = False
            logger.info("Knowledge update scheduler stopped")

    def add_job(
        self,
        job_id: str,
        func: Callable,
        trigger: str = 'interval',
        **kwargs
    ) -> None:
        if not self.scheduler:
            raise RuntimeError("Scheduler not started")

        try:
            if trigger == 'interval':
                trigger_obj = IntervalTrigger(**kwargs)
            elif trigger == 'cron':
                trigger_obj = CronTrigger(**kwargs)
            else:
                raise ValueError(f"Unsupported trigger type: {trigger}")

            self.scheduler.add_job(
                func=func,
                trigger=trigger_obj,
                id=job_id,
                replace_existing=True
            )

            self.jobs[job_id] = {
                'function': func.__name__,
                'trigger': trigger,
                'kwargs': kwargs,
                'added_at': datetime.now()
            }

            logger.info(
                "Job added to scheduler",
                job_id=job_id,
                function=func.__name__,
                trigger=trigger
            )

        except Exception as e:
            logger.error("Failed to add job", job_id=job_id, error=str(e))
            raise

    def remove_job(self, job_id: str) -> None:
        if not self.scheduler:
            raise RuntimeError("Scheduler not started")

        try:
            self.scheduler.remove_job(job_id)
            self.jobs.pop(job_id, None)
            logger.info("Job removed from scheduler", job_id=job_id)
        except Exception as e:
            logger.error("Failed to remove job", job_id=job_id, error=str(e))
            raise

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        if job_id not in self.jobs:
            return None

        if not self.scheduler:
            return self.jobs[job_id]

        job = self.scheduler.get_job(job_id)
        if job:
            return {
                **self.jobs[job_id],
                'next_run': job.next_run_time,
                'is_scheduled': job.next_run_time is not None
            }
        return self.jobs[job_id]

    def list_jobs(self) -> List[Dict[str, Any]]:
        jobs = []
        for job_id in self.jobs:
            status = self.get_job_status(job_id)
            if status:
                jobs.append(status)
        return jobs

    def _schedule_periodic_updates(self) -> None:
        scheduler_config = self.config.get('scheduler', {})
        update_interval = scheduler_config.get('update_interval_hours', 24)

        try:
            from ..data_sources.manager import DataSourceManager
            from ..vector_db.manager import VectorDBManager

            def update_knowledge_base():
                logger.info("Starting scheduled knowledge base update")

                try:
                    if self._data_manager is None:
                        self._data_manager = DataSourceManager()
                    if self._vector_manager is None:
                        self._vector_manager = VectorDBManager()

                    new_data = self._data_manager.fetch_all_sources()

                    if new_data:
                        processed_data = self._data_manager.process_data(new_data)
                        self._vector_manager.update_knowledge_base(processed_data)

                        logger.info(
                            "Knowledge base update completed",
                            articles_processed=len(processed_data),
                            sources_updated=len(new_data)
                        )
                    else:
                        logger.info("No new data to process")

                except Exception as e:
                    logger.error("Knowledge base update failed", error=str(e))
                    raise

            self.add_job(
                job_id='knowledge_base_update',
                func=update_knowledge_base,
                trigger='interval',
                hours=update_interval
            )

            logger.info(f"Scheduled periodic updates every {update_interval} hours")

        except ImportError as e:
            logger.warning("Could not schedule updates due to missing modules", error=str(e))
        except Exception as e:
            logger.error("Failed to schedule periodic updates", error=str(e))
            raise

    def is_running(self) -> bool:
        return self._running and self.scheduler is not None

    def get_next_run_time(self, job_id: str) -> Optional[datetime]:
        if not self.scheduler:
            return None

        job = self.scheduler.get_job(job_id)
        return job.next_run_time if job else None

    def pause_job(self, job_id: str) -> None:
        if not self.scheduler:
            raise RuntimeError("Scheduler not started")

        try:
            self.scheduler.pause_job(job_id)
            logger.info("Job paused", job_id=job_id)
        except Exception as e:
            logger.error("Failed to pause job", job_id=job_id, error=str(e))
            raise

    def resume_job(self, job_id: str) -> None:
        if not self.scheduler:
            raise RuntimeError("Scheduler not started")

        try:
            self.scheduler.resume_job(job_id)
            logger.info("Job resumed", job_id=job_id)
        except Exception as e:
            logger.error("Failed to resume job", job_id=job_id, error=str(e))
            raise


_scheduler: Optional[KnowledgeUpdateScheduler] = None


def get_scheduler() -> KnowledgeUpdateScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = KnowledgeUpdateScheduler()
    return _scheduler


def start_scheduler() -> None:
    get_scheduler().start()


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler:
        _scheduler.stop()