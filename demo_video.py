import argparse
import cv2
import torch

from model import SCNN
from model_ENET_SAD import ENet_SAD
from utils.prob2lines import getLane
from utils.transforms import *

import time
from multiprocessing import Process, JoinableQueue, SimpleQueue
from threading import Lock

img_size = (640, 360)
#net = SCNN(input_size=(800, 288), pretrained=False)
net = ENet_SAD(img_size, sad=False)
# CULane mean, std
mean=(0.3598, 0.3653, 0.3662)
std=(0.2573, 0.2663, 0.2756)
# Imagenet mean, std
# mean=(0.485, 0.456, 0.406)
# std=(0.229, 0.224, 0.225)
transform_img = Resize(img_size)
transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))

pipeline = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", '-i', type=str, default="demo/d193a8a5-9df343d6.mov", help="Path to demo video")
    parser.add_argument("--weight_path", '-w', type=str, default="experiments/exp2_BDD100K_ENet+ENet-SAD/exp2_best.pth", help="Path to model weights")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    args = parser.parse_args()
    return args


def network(net, img):
    seg_pred, exist_pred = net(img.cuda())[:2]
    seg_pred = seg_pred.detach().cpu()
    exist_pred = exist_pred.detach().cpu()
    return seg_pred, exist_pred


def visualize(img, seg_pred, exist_pred):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lane_img = np.zeros_like(img)
    color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
    coord_mask = np.argmax(seg_pred, axis=0)
    for i in range(0, 4):
        if exist_pred[0, i] > 0.5:
            lane_img[coord_mask == (i + 1)] = color[i]
    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
    return img


def pre_processor(arg):
    img_queue, video_path = arg
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        if img_queue.empty():
            ret, frame = cap.read()
            if ret:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = transform_img({'img': frame})['img']
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x = transform_to_net({'img': img})['img']
                x.unsqueeze_(0)

                img_queue.put(x)
                img_queue.join()
            else:
                break


def post_processor(arg):
    img_queue, arg_visualize = arg

    while True:
        if not img_queue.empty():
            x, seg_pred, exist_pred = img_queue.get()
            seg_pred = seg_pred.numpy()[0]
            exist_pred = exist_pred.numpy()

            exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

            print(exist)
            for i in getLane.prob2lines_CULane(seg_pred, exist):
                print(i)

            if arg_visualize:
                frame = x.squeeze().permute(1, 2, 0).numpy()
                img = visualize(frame, seg_pred, exist_pred)
                cv2.imshow('input_video', frame)
                cv2.imshow("output_video", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            pass


def main():
    args = parse_args()
    video_path = args.video_path
    weight_path = args.weight_path

    if pipeline:
        input_queue = JoinableQueue()
        pre_process = Process(target=pre_processor, args=((input_queue, video_path),))
        pre_process.start()

        output_queue = SimpleQueue()
        post_process = Process(target=post_processor, args=((output_queue, args.visualize),))
        post_process.start()
    else:
        cap = cv2.VideoCapture(video_path)

    save_dict = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(save_dict['net'])
    net.eval()
    net.cuda()

    while True:
        if pipeline:
            loop_start = time.time()
            x = input_queue.get()
            input_queue.task_done()

            gpu_start = time.time()
            seg_pred, exist_pred = network(net, x)
            gpu_end = time.time()

            output_queue.put((x, seg_pred, exist_pred))

            loop_end = time.time()

        else:
            if not cap.isOpened():
                break

            ret, frame = cap.read()

            if ret:
                loop_start = time.time()
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = transform_img({'img': frame})['img']
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                x = transform_to_net({'img': img})['img']
                x.unsqueeze_(0)

                gpu_start = time.time()
                seg_pred, exist_pred = network(net, x)
                gpu_end = time.time()

                seg_pred = seg_pred.numpy()[0]
                exist_pred = exist_pred.numpy()

                exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

                print(exist)
                for i in getLane.prob2lines_CULane(seg_pred, exist):
                    print(i)

                loop_end = time.time()

                if args.visualize:
                    img = visualize(img, seg_pred, exist_pred)
                    cv2.imshow('input_video', frame)
                    cv2.imshow("output_video", img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        print("gpu_runtime:", gpu_end - gpu_start, "FPS:", int(1 / (gpu_end - gpu_start)))
        print("total_runtime:", loop_end - loop_start, "FPS:", int(1 / (loop_end - loop_start)))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
