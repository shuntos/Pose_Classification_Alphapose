import os
import time
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms, write_json

DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (640, 480)
}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def calculate_angle(point1, point2):
    # Convert points to NumPy arrays
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the vector between the two points
    vector = point2 - point1

    # Calculate the angle using the arctan2 function
    angle_rad = np.arctan2(vector[1], vector[0])
    angle_deg = np.degrees(angle_rad)

    # Ensure the angle is between 0 and 360 degrees
    angle_deg = (angle_deg + 360) % 360

    return angle_deg



def classify_pose(points):
    left_hip_knee  = [12, 14]
    right_hip_knee = [11, 13]

    left_knee_ankle  = [14, 16]
    right_knee_ankle = [13, 15]

    angle_left_hip = calculate_angle(points[left_hip_knee[0]], points[left_hip_knee[1]])
    angle_right_hip = calculate_angle(points[right_hip_knee[0]], points[right_hip_knee[1]])

    angle_left_knee = calculate_angle(points[left_knee_ankle[0]], points[left_knee_ankle[1]])
    angle_right_knee = calculate_angle(points[right_knee_ankle[0]], points[right_knee_ankle[1]])

    pose_label = "None"

    print("Angle between left hip and knee", angle_left_hip,"\nAngle between Right hip and knee", angle_right_hip)

    print("Angle left knee and ankle", angle_left_knee, "\nAngle right knee and ankle", angle_right_knee)

    if angle_right_hip > 60  and angle_right_hip < 120 and angle_left_hip > 60 and angle_left_hip < 120:
        print("Standing postion")

        pose_label = "Standing"

    elif angle_right_hip <60  or angle_right_hip > 120 and angle_left_hip < 60 or angle_left_hip >120: # Check if angle between hip and knee tends to horizonal

        if angle_left_knee > 60  and angle_left_knee < 120 and angle_right_knee > 60 and angle_right_knee < 120: # Angle tends to 90 degree vertical
            print("Sitting")
            pose_label = "Sitting"

        else:
            print("Lying")
            pose_label = "Laying"

    return pose_label 

class DataWriter():
    def __init__(self, cfg, opt, save_video=False,
                 video_save_opt=DEFAULT_VIDEO_SAVE_OPT,
                 queueSize=1024):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt

        self.eval_joints = EVAL_JOINTS
        self.save_video = save_video
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))

        if self.opt.save_img or self.save_video or self.opt.vis:
            loss_type = self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss')
            num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
            if loss_type == 'MSELoss':
                self.vis_thres = [0.4] * num_joints
            elif 'JointRegression' in loss_type:
                self.vis_thres = [0.05] * num_joints
            elif loss_type == 'Combined':
                if num_joints == 68:
                    hand_face_num = 42
                else:
                    hand_face_num = 110
                self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num

        self.use_heatmap_loss = (self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def update(self):
        final_result = []
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE

        print("Save video path========",self.video_save_opt["savepath"] )
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            if not stream.isOpened():
                print("Try to use other video encoders...")
                ext = self.video_save_opt['savepath'].split('.')[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt['fourcc'] = fourcc
                self.video_save_opt['savepath'] = self.video_save_opt['savepath'][:-4] + _ext
                stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            assert stream.isOpened(), 'Cannot open video for writing'
        # keep looping infinitelyd

        data_list = [] 
        npy_file_path = "/content/gdrive/MyDrive/AlphaPose/dataset/test_array/"

        if not os.path.exists(npy_file_path):
            os.makedirs(npy_file_path)


        while True:
            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.wait_and_get(self.result_queue)
            if orig_img is None:
                # if the thread indicator variable is set (img is None), stop the thread
                if self.save_video:
                    stream.release()
                write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                print("Results have been written to json.")
                return
            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            if boxes is None or len(boxes) == 0:
                if self.opt.save_img or self.save_video or self.opt.vis:
                    self.write_image(orig_img, im_name, stream=stream if self.save_video else None)
            else:
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                assert hm_data.dim() == 4

                face_hand_num = 110
                if hm_data.size()[1] == 136:
                    self.eval_joints = [*range(0,136)]
                elif hm_data.size()[1] == 26:
                    self.eval_joints = [*range(0,26)]
                elif hm_data.size()[1] == 133:
                    self.eval_joints = [*range(0,133)]
                elif hm_data.size()[1] == 68:
                    face_hand_num = 42
                    self.eval_joints = [*range(0,68)]
                elif hm_data.size()[1] == 21:
                    self.eval_joints = [*range(0,21)]
                pose_coords = []
                pose_scores = []
                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    if isinstance(self.heatmap_to_coord, list):
                        pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                            hm_data[i][self.eval_joints[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type)
                        pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                            hm_data[i][self.eval_joints[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                        pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                        pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
                    else:
                        pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)
                if not self.opt.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

                _result = []
                for k in range(len(scores)):

                    _result.append(
                        {
                            'keypoints':preds_img[k],
                            'kp_score':preds_scores[k],
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                            'idx':ids[k],
                            'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                        }

                    )


                    keypoints_pose = preds_img[k].numpy().astype(int)
                    print("Image:",im_name)

                    pose_label = classify_pose(keypoints_pose)

                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (100,100)
                    fontScale              = 2
                    fontColor              = (0,255,0)
                    thickness              = 2
                    lineType               = 2

                  
                    npy_path = npy_file_path+im_name[:-4]+".npy"
                    np.save(npy_path, np.array(keypoints_pose))



                result = {
                    'imgname': im_name,
                    'result': _result
                }


                if self.opt.pose_flow:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']

                final_result.append(result)
                if self.opt.save_img or self.save_video or self.opt.vis:
                    if hm_data.size()[1] == 49:
                        from alphapose.utils.vis import vis_frame_dense as vis_frame
                    elif self.opt.vis_fast:
                        from alphapose.utils.vis import vis_frame_fast as vis_frame
                    else:
                        from alphapose.utils.vis import vis_frame
                    img = vis_frame(orig_img, result, self.opt, self.vis_thres)


                    cv2.putText(img,pose_label, 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

                    self.write_image(img, im_name, stream=stream if self.save_video else None)




    def write_image(self, img, im_name, stream=None):
        if self.opt.vis:
            cv2.imshow("AlphaPose Demo", img)
            cv2.waitKey(30)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
        if self.save_video:
            stream.write(img)

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

    def clear_queues(self):
        self.clear(self.result_queue)
        
    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    def recognize_video_ext(self, ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'
