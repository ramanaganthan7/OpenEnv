"""
ClimateWatch — Task Registry
"""
from app.tasks.task1_detect  import load_task1, grade_task1
from app.tasks.task2_clean   import load_task2, grade_task2
from app.tasks.task3_cascade import load_task3, grade_task3

TASK_LOADERS = {
    "task1_detect":  load_task1,
    "task2_clean":   load_task2,
    "task3_cascade": load_task3,
}

TASK_GRADERS = {
    "task1_detect":  grade_task1,
    "task2_clean":   grade_task2,
    "task3_cascade": grade_task3,
}


def get_grader(task_id: str):
    return TASK_GRADERS[task_id]


def get_loader(task_id: str):
    return TASK_LOADERS[task_id]
