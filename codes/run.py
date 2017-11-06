import projection
from video import Video
from sim import Sim

if __name__ == '__main__':
    # video_table = VideoTable("test_data/check0.png")
    # print(video_table.field_corners)

    video_file = "test_data/game1/clip1.mp4"
    # video_file = "test_data/game1/clip2.mp4"
    # video_file = "test_data/game1/clip3.mp4"
    # video_file = "test_data/game2/clip4.mp4"

    my_video1 = Video(video_file, resize=True)
    my_video1.real_time_tracking()
    my_video1.draw_simple_trajectory()

    # print(my_video1.ball_tracking_rec_complete)

    my_sim = Sim()
    H = projection.create_homography(my_video1.table.table_corners, my_sim.sim_table.table_corners)
    print(H)

    sim_tracking_ball_dic = projection.gen_tracking_ball_dic_from_video(my_video1.ball_tracking_rec_complete, H)
    # print(sim_tracking_ball_dic)

    my_sim.set_tracking_ball_dic(sim_tracking_ball_dic)
    my_sim.draw_simple_trajectory()
    my_sim.animation()





