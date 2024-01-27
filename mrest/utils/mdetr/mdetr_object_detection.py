import torch
from PIL import Image
import requests
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from omegaconf import OmegaConf
from typing import Any, Mapping, Optional

from mrest.utils.env_helpers import (read_config_for_parameterized_envs, filter_train_val_task_names)


def load_model_mdetr():
    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_resnet101',
                                           pretrained=True, return_postprocessor=True)
    model = model.cuda()
    breakpoint()
    model.eval()
    return model

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]



# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def plot_results(pil_img, scores, boxes, labels, masks=None, save=False,
                 save_dir: Optional[str] = None,):
    plt.figure(figsize=(16, 8))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))
    
    if save:
        plt.imshow(np_image)
        if save_dir is None:
            save_dir = './image_with_bb.png'
        plt.savefig(save_dir)

def bbox_inference(model, im, caption):
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0).cuda()

    memory_cache = model(img, [caption], encode_and_save=True)
    breakpoint()
    outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

    # keep only predictions with 0.7+ confidence
    probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
    keep = (probas > 0.7).cpu()
    # keep = (probas == max(probas))

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

    # Extract the text spans predicted by each box
    positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
    predicted_spans = defaultdict(str)
    for tok in positive_tokens:
        item, pos = tok
        if pos < 255:
            span = memory_cache["tokenized"].token_to_chars(0, pos)
            predicted_spans [item] += " " + caption[span.start:span.end]

    labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]

    # plot_results(im, probas[keep], bboxes_scaled, labels)
    return probas[keep], bboxes_scaled, labels

def make_heatmap(im, bboxes_scaled, save=False, save_dir='./heatmap.png'):
    import cv2
    plot_goal = bboxes_scaled.shape[0] > 1

    if len(bboxes_scaled) == 0:
        width, height = im.size
        return np.zeros((height, width))

    xmin, ymin, xmax, ymax = bboxes_scaled[0]
    # import ipdb; ipdb.set_trace()
    target_pixel_x = int((xmin.item()+xmax.item())/2)
    target_pixel_y = int((ymin.item()+ymax.item())/2)
    if plot_goal:
        xmin, ymin, xmax, ymax = bboxes_scaled[1]
        goal_pixel_x = int((xmin.item()+xmax.item())/2)
        goal_pixel_y = int((ymin.item()+ymax.item())/2)

    width, height = im.size
    heatmap = np.zeros((height, width))
    heatmap[target_pixel_y, target_pixel_x] = 1.
    if plot_goal:
        heatmap[goal_pixel_y, goal_pixel_x] = 0.5

    if save:
        cv2.imwrite(str(save_dir),heatmap*255)
    
    return heatmap*255


def save_object_detection_heatmaps(data_dir, env_type, num_demos_per_task, camera_names):
    model = load_model_mdetr()

    env_config_dict_by_type = read_config_for_parameterized_envs(data_dir, read_all_configs=True)
    env_config_dict = env_config_dict_by_type[env_type]
    task_names = filter_train_val_task_names(env_config_dict, None)
    data_demo_idxs = np.arange(num_demos_per_task)
    image_names_by_task = dict()

    num_tasks = len(task_names)
    for task_index, task_name in enumerate(task_names):
        print(f"Heatmap for {task_index}/{num_tasks} task")
        task_dir = Path(data_dir) / env_type / task_name
        image_names_by_task[task_name] = dict()
        skill = env_config_dict[task_name]['task_command_type']
        block_color = env_config_dict[task_name]['blockA_config']['color'].split('block_')[1]
        task_desc_obj = f'{block_color} small block'
        
        for demo_idx, demo_dir in enumerate(task_dir.iterdir()):
            assert demo_dir.name.startswith('demo')
            if demo_idx in data_demo_idxs:
                image_names_by_task[task_name][demo_dir] = dict()
                for camera_name in camera_names:
                    img_dir = demo_dir / camera_name
                    
                    heatmap_dir = demo_dir / 'heatmap'
                    bbox_dir = demo_dir / 'bbox'
                    
                    demo_images = [f for f in img_dir.iterdir() if f.suffix == '.png']
                    demo_images = sorted(demo_images, key=lambda x: int(x.name.split('img_t')[1].split('.')[0]))

                    img_name = demo_images[0].name.split('.png')[0]
                    heatmap_image_name = heatmap_dir / f'{img_name}_mdetr.png'
                    bbox_image_name = bbox_dir / f'{img_name}_mdetr.png'
                    
                    if not heatmap_dir.exists():
                        heatmap_dir.mkdir()
                    if not bbox_dir.exists():
                        bbox_dir.mkdir()
                    
                    image_names_by_task[task_name][demo_dir][camera_name] = demo_images
                    im = Image.open(demo_images[0]) # get first image of trajectory

                    if 'place' in skill:
                        goal_caption = 'grey sphere'
                    elif 'stack' in skill:
                        block_color = env_config_dict[task_name]['blockB_config']['color'].split('block_')[1]
                        goal_caption = f'{block_color} small block'
                    else:
                        goal_caption = None
                    
                    probas, bboxes_scaled, labels = bbox_inference(model, im, task_desc_obj)
                    if goal_caption is not None:
                        probas_goal, bb_goal, labels_goal = bbox_inference(model, im, goal_caption)
                        
                        probas = torch.cat((probas, probas_goal))
                        bboxes_scaled = torch.cat((bboxes_scaled, bb_goal), dim=0)
                        labels = labels + labels_goal
                    
                    plot_results(im, probas, bboxes_scaled, labels, save=True, save_dir=bbox_image_name)

                    heatmap = make_heatmap(im, bboxes_scaled, save=True, save_dir=heatmap_image_name)

def object_mask_mdetr(model, im, skill, task_desc, save=False, use_skill_desc: bool = False,
                      save_dir: Optional[str] =  None,):

    # if isinstance(im, Image.Image):
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)

    block_color = task_desc.split(' block')[0].split(' ')[-1]
    task_desc_obj = f'{block_color} small block'

    if use_skill_desc:
        if 'place' in skill:
            goal_caption = 'grey sphere'
        elif 'stack' in skill:
            block_color = task_desc.split(' block')[1].split(' ')[-1]
            goal_caption = f'{block_color} small block'
        else:
            goal_caption = None
    else:
        goal_caption = None

    probas, bboxes_scaled, labels = bbox_inference(model, im, task_desc_obj)
    if goal_caption is not None:
        probas_goal, bb_goal, labels_goal = bbox_inference(model, im, goal_caption)
        
        probas = torch.cat((probas, probas_goal))
        bboxes_scaled = torch.cat((bboxes_scaled, bb_goal), dim=0)
        labels = labels + labels_goal

    if save:
        plot_results(im, probas, bboxes_scaled, labels, save=save, save_dir=save_dir)
    heatmap = make_heatmap(im, bboxes_scaled, save=save)
    return heatmap


def run_on_images_config(model, config: Mapping[str, Any], save_dir: str):
    # save_dir = '/home/mohit/experiment_results/object_centric/real_world/object_detection/output_mdetr_default'
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    for img_idx, img_cfg in config.items():
        img_path = img_cfg['path']
        lang_descriptions = img_cfg['lang']
        image = np.array(Image.open(img_path))
        if image.shape[-1] > 3:
            image = image[:, :, :3]
        for lang_idx, desc in enumerate(lang_descriptions):
            filepath = save_dir / f"img_{img_idx:04d}" / f"pred_lang_{lang_idx:01d}_{desc.replace(' ', '_')}.png"

            if not filepath.parent.exists():
                filepath.parent.mkdir(parents=True)

            heatmap = object_mask_mdetr(model, image, 'stack', desc, save=True,
                                        use_skill_desc=False, save_dir=filepath)


def load_object_detection_config():
    abs_path = Path('/home/mohit/projects/object_centric/r3m/evaluation/r3meval')
    rel_path = Path('core/config/object_detect_with_lang_config.yaml')
    config = OmegaConf.load(abs_path / rel_path)
    return config

def main():
    config = load_object_detection_config()
    model = load_model_mdetr()
    path_to_save = '/home/mohit/experiment_results/object_centric/real_world/object_detection/output_mdetr_default'
    run_on_images_config(model, config, path_to_save)


if __name__ == "__main__":
    # data_dir = "/home/saumyas/experiment_results/object_centric/r3m/data/sawyer_mt_gen_multicolor_multiview_block_noise_with_correct_action_multiskill_target"
    # env_type = 'train'
    # num_demos_per_task = 4
    # camera_names = ['left_cap2']

    # # url = "http://images.cocodataset.org/val2017/000000281759.jpg"
    # # im = Image.open(requests.get(url, stream=True).raw)
    # # plot_inference(im, "5 people each holding an umbrella")
    
    # img_dir = "/home/saumyas/experiment_results/object_centric/r3m/data/sawyer_mt_gen_multicolor_multiview_block_noise_with_correct_action_multiskill_target/train/env_stack_target_block_blue_block_black_idx_46/demo0/left_cap2/img_t0.png"
    # image = np.array(Image.open(img_dir))
    # text = "Stack blue block on black block"
    # model = load_model_mdetr()
    # heatmap = object_mask_mdetr(model, image, 'stack', text)
    
    # save_object_detection_heatmaps(data_dir, env_type, num_demos_per_task, camera_names)

    run_config = True
    if run_config: 
        main()
    else:
        img_dir = "/home/saumyas/Pictures/train_test.png"
        image = Image.open(img_dir)
        caption = "Blue men shoe. Green men shoe. Black men shoe. Women shoe"
        model = load_model_mdetr()
        probas, bboxes_scaled, labels = bbox_inference(model, image, caption)
        plot_results(image, probas, bboxes_scaled, labels, save=True, save_dir='./train_test.jpg')
    
