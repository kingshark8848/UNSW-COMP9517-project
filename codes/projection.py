import numpy as np
import cv2


def create_homography(video_keypoints, sim_keypoints):

    hgcoord_filepath = 'hgmatrix.txt'

    H = cv2.findHomography(video_keypoints, sim_keypoints)[0]
    np.savetxt(hgcoord_filepath, H)
    return H


def gen_tracking_ball_dic_from_video(ball_rec_video, H):
    """
    :param ball_rec_video: # {"0":[(111,222),(111,333),None,...], "3":..., ...}
    :param H:
    :return:
    """

    ball_rec_sim = {}
    for ball_id in ball_rec_video:
        ball_rec_sim[ball_id] = []

        for p_video in ball_rec_video[ball_id]:

            if not p_video:
                ball_rec_sim[ball_id].append(None)
                continue

            p_video = np.array([p_video], dtype='float32')
            p_video = np.array([p_video])

            p_sim = cv2.perspectiveTransform(p_video, H)

            p_sim = (p_sim[0][0][0], p_sim[0][0][1])
            ball_rec_sim[ball_id].append(p_sim)

    return ball_rec_sim
