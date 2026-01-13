from per_segment_anything import SamPredictor
from evaluation.evaluate_2d import eval_fundus
from evaluation.evaluate_3d import eval_3d

def evaluate_main(sam,config,is_train=True):
    sam.eval().cuda()
    sam.MedSAM_norm = False
    predictor = SamPredictor(sam)
    ret_dict ={}

    task_dict = config.get('evaluation', {}).get('tasks', [])
    tasks = [task['task'] for task in task_dict]
    task_weights = [task['weight'] for task in task_dict if 'weight' in task.keys()]

    data_pth = config['evaluation']['data_path']
    for task,weight  in zip(tasks,task_weights):
        if 'fundus' in task.lower():
            res = eval_fundus(predictor, task.lower(), data_pth, is_train)
        else:
            # for 3d
            res = eval_3d(predictor, task, data_pth, is_train)
        print(f"{task}: {res:.3f}")
        ret_dict[task] = res
    return ret_dict