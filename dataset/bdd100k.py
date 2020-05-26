import json
import os
# import logging
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm


# from .Bezier import Bezier

class bdd100k(Dataset):
    """
    path = D:\bdd100k_images\bdd100k\images\100k
    splitted in to three partitions: train, val, test.
    labels for json : D:\bdd100k_labels_release\bdd100k\labels
    \bdd100k_labels_images_train.json, bdd100k_labels_images_val.json
    """

    def __init__(self, path, image_set, transforms=None):  # image_set : 'train', 'val', 'test'
        super(bdd100k, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.transforms = transforms

        self.createIndex_train()

    def createIndex_train(self):

        self.img_list = []
        self.segLabel_list = []

        if self.image_set == "train":  # 받은 값, txt로 저장
            self.image_set2 = "train"  # 참고할 json이름
            self.image_set3 = "train"  # txt 읽는 값
        elif self.image_set == "val":
            self.image_set2 = "train"
            self.image_set3 = "train"
        elif self.image_set == "test":
            self.image_set2 = "val"
            self.image_set3 = "test"

        if not os.path.isfile(
                os.path.join(self.data_dir_path, "seg_label", "list", "{}_gt.txt".format(self.image_set3))):
            self._gen_label_for_json()  # self 넣으면 안됨

        f = open(os.path.join(self.data_dir_path, "seg_label", "list", "{}_gt.txt".format(self.image_set3)))
        self.segLabel_list = f.read().splitlines()
        # self.segLabel_list = f.readlines()[:-3]

        random.seed(2020)
        if self.image_set == "train":
            self.segLabel_list = random.sample(self.segLabel_list, int(len(self.segLabel_list) * 0.8))
        elif self.image_set == "val":
            temp = random.sample(self.segLabel_list, int(len(self.segLabel_list) * 0.8))
            self.segLabel_list = list(set(self.segLabel_list) - set(temp))

        for i in range(len(self.segLabel_list)):
            a = self.segLabel_list[i][-21:-3]
            a = a + "jpg"  ############ string은 append 대신 +
            self.img_list.append(os.path.join(self.data_dir_path, "images", "100k", self.image_set3, a))

        print(self.image_set)
        print(len(self.img_list))

        '''
        img = cv2.imread(self.img_list[0])
        cv2.imshow("dhdvi", img)
        cv2.waitKey(0)
        h, w, c = img.shape
        print(h, w, c)

        img = cv2.imread(self.segLabel_list[0])
        cv2.imshow("dhdvi", img)
        cv2.waitKey(0)
        h, w, c = img.shape
        print(h, w, c)  #왜 세그라벨 오류 ....
        '''

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow("jdifj", img)
        # cv2.waitKey(0)

        if self.image_set2 != 'test':
            segLabel = cv2.imread(self.segLabel_list[idx], cv2.COLOR_BGR2GRAY)
            exist = np.array([1, 1, 1, 1])
        else:
            segLabel = None
            exist = None

        sample = {'img': img,
                  'segLabel': segLabel,
                  'exist': exist,
                  'img_name': self.img_list[idx]}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self):
        # print(self.img_list)
        return len(self.img_list)

    def _gen_label_for_json(self):
        H, W = 720, 1280
        SEG_WIDTH = 8 * 2 #논문에 8로 나와있고, 모델에서 사용하기 위해 이미지 크기 절반으로 줄이므로
        t_points = np.arange(0, 1, 0.1)
        json_dir = os.path.join(self.data_dir_path, "labels")
        save_dir = "seg_label"

        # data_dir_path : "D:\\bdd100k_images\\bdd100k\\images\\100k"
        os.makedirs(os.path.join(self.data_dir_path, save_dir, "list"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir_path, save_dir, self.image_set), exist_ok=True)
        list_f = open(os.path.join(self.data_dir_path, save_dir, "list", "{}_gt.txt".format(self.image_set)), "w")

        # with open(os.path.join(json_dir, 'bdd100k','labels',"test_json.json"), "r", newline="\n") as outfile: #.format(self.image_set)
        with open(os.path.join(json_dir, "bdd100k_labels_images_{}.json".format(self.image_set2)), "r",
                  newline="\n") as outfile:
            data = json.load(outfile)

            progressbar = tqdm(range(len(data)))

            for json_name in data:  # each images
                vertices = []
                types = []  # LL, CCCL
                type = []  # 0, 1
                laneDirctions = []
                laneDirction = []
                for i in json_name["labels"]:  # image 안의 라벨들
                    if i["category"] == "lane":
                        for j in i["poly2d"]:
                            vertices.append(j["vertices"])
                            types.append(j["types"])
                            closed = j["closed"]
                        laneDirctions.append(i["attributes"]["laneDirection"])
                for i in types:
                    typee = []
                    for j in range(len(i)):
                        if i[j] == "L":
                            typee.append(0)
                        elif i[j] == "C":
                            typee.append(1)  # bezier curve
                            j += 3
                    type.append(typee)
                for i in laneDirctions:
                    if i == "parallel":
                        laneDirction.append(1)  # 1 : parallel
                    elif i == "vertical":
                        laneDirction.append(2)  # 2 : vertical
                # print(json_name["name"]) ###################################
                img_name = json_name["name"]
                # print(vertices) ###################### 좌표 필요
                # print(type)    ##################### 0 : line, 1 : bezier
                # print(laneDirction) ######################## 1 : parallel, 2 : vertical
                # print("----------------")

                seg_img = np.zeros((H, W, 1))
                list_str = []

                for i, j, k in zip(vertices, type, laneDirction):  # vertices : segment에 필요한 점들의 총 집합
                    points = []
                    cvpoints = []
                    for point in i:  # i : 한 line이나 곡선을 그리는데 필요한 점들
                        points.append(point)
                    # print("points", points) #각 좌표 필요
                    count = 0
                    for x in range(len(j) - 1):
                        # print("range(len(j)-1)", range(len(j)-1))
                        # print("j", j)
                        # print("j[x]", j[x])
                        # print("x", x)
                        if count:
                            count -= 1
                            continue
                        if j[x] == 0:
                            # cvpoints.append( (int(points[x][0]),int(points[x][1])) )
                            cv2.line(seg_img, (int(points[x][0]), int(points[x][1])),
                                     (int(points[x + 1][0]), int(points[x + 1][1])), (k, k, k), SEG_WIDTH // 2)

                        elif j[x] == 1:
                            count = 3
                            a = [[int(points[x - 3][0]), int(points[x - 3][1])],
                                 [int(points[x - 2][0]), int(points[x - 2][1])],
                                 [int(points[x - 1][0]), int(points[x - 1][1])], [int(points[x][0]), int(points[x][1])]]
                            a = np.array(a)
                            line = Bezier.Curve(t_points, a)  # numpy.ndarray
                            line = line.tolist()
                            line.append([points[3][0], points[3][1]])
                            # cvpoints.append(line)
                            for y in range(len(line) - 1):
                                cv2.line(seg_img, (int(line[y][0]), int(line[y][1])),
                                         (int(line[y + 1][0]), int(line[y + 1][1])), (k, k, k), SEG_WIDTH // 2)

                seg_path = os.path.join(self.data_dir_path, save_dir, self.image_set, img_name[:-3] + "png")
                # if seg_path[0] != '/':
                #    seg_path = '/' + seg_path
                list_str.insert(0, seg_path)
                list_str = " ".join(list_str) + "\n"
                list_f.write(list_str)
                # print(seg_path)
                # seg_path = os.path.join(seg_path, str(y)+".png")
                cv2.imwrite(seg_path, seg_img)
                # print("===============")
                progressbar.update(1)
            progressbar.close()

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
            exist = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
            exist = torch.stack([b['exist'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]
            exist = [b['exist'] for b in batch]

        samples = {'img': img,
                   'segLabel': segLabel,
                   'exist': exist,
                   'img_name': [x['img_name'] for x in batch]}

        return samples


class Bezier():
    def TwoPoints(t, P1, P2):
        if not isinstance(P1, np.ndarray) or not isinstance(P2, np.ndarray):
            raise TypeError('Points must be an instance of the numpy.ndarray!')
        if not isinstance(t, (int, float)):
            raise TypeError('Parameter t must be an int or float!')
        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def Points(t, points):
        newpoints = []
        for i1 in range(0, len(points) - 1):
            newpoints += [Bezier.TwoPoints(t, points[i1], points[i1 + 1])]
        return newpoints

    def Point(t, points):
        newpoints = points
        while len(newpoints) > 1:
            newpoints = Bezier.Points(t, newpoints)
        return newpoints[0]

    def Curve(t_values, points):
        if not hasattr(t_values, '__iter__'):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if len(t_values) < 1:
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")
        if not isinstance(t_values[0], (int, float)):
            raise TypeError("`t_values` Must be an iterable of integers or floats, of length greater than 0 .")

        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            curve = np.append(curve, [Bezier.Point(t, points)], axis=0)
        curve = np.delete(curve, 0, 0)
        return curve


'''
def main():
    c = bdd100k("D:\\bdd100k_images\\bdd100k\\images\\100k",'val')

if __name__ == "__main__":
    main()
'''