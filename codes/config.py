video_table_data = {
    "hsv_color_lower": (95, 100, 0),
    "hsv_color_upper": (105, 255, 255)
}

balls_data = {
    "0":
        {"hsv_color_lower": (0, 10, 100),
         "hsv_color_upper": (40, 80, 255),
         "trajectory_color": (255, 255, 255)
         },
    "2":
        {"hsv_color_lower": (99, 10, 50),
         "hsv_color_upper": (119, 255, 150),
         "trajectory_color": (112, 77, 49)
         },
    "3":
        {"hsv_color_lower": (3,150,150),
         "hsv_color_upper": (15,230,255),
         "trajectory_color": (56, 144, 249)
         },
    "4":
        {"hsv_color_lower": (0, 100, 150),
         "hsv_color_upper": (15, 190, 255),
         "trajectory_color": (254, 102, 255)
         },
    "5":
        {"hsv_color_lower": (0, 130, 150),
         "hsv_color_upper": (12, 255, 255),
         "trajectory_color": (37, 61, 231)
         },
    "6":
        {"hsv_color_lower": (78, 50, 50),
         "hsv_color_upper": (100, 255, 255),
         "trajectory_color": (55, 58, 13)
         },
    "7":
        {"hsv_color_lower": (12, 130, 130),
         "hsv_color_upper": (25, 200, 200),
         "trajectory_color": (66, 129, 175)
         },
    "8":
        {"hsv_color_lower": (0, 50, 0),
         "hsv_color_upper": (255, 255, 50),
         "trajectory_color": (0, 0, 0)
         },
    "9":
        {"hsv_color_lower": (10, 100, 180),
         "hsv_color_upper": (28, 185, 255),
         "trajectory_color": (73, 178, 243)
         }
}

sim_ball_data = {
    "0":
        {"ball_color": (255, 255, 255), "ball_image": None},
    "3":
        {"ball_color": (56, 144, 249), "ball_image": None},
    "2":
        {"ball_color": (112, 77, 49), "ball_image": None},
    "4":
        {"ball_color": (254, 102, 255), "ball_image": None},
    "5":
        {"ball_color": (37, 61, 231), "ball_image": None},
    "6":
        {"ball_color": (55, 58, 13), "ball_image": None},
    "7":
        {"ball_color": (66, 129, 175), "ball_image": None},
    "8":
        {"ball_color": (0, 0, 0), "ball_image": None},
    "9":
        {"ball_color": (73, 178, 243), "ball_image": None}
}

first_frame_save_path = "test_data/first_frame.png"

video_ball_radius = 10
sim_ball_radius = 10

max_move_dis = 100

black_v_max = 10

white_s_max = 10

mean_hue_variance = 10

balls_avg_BGR = {
    "0": (255, 255, 255),  # This is used to match write ball or stripped balls
    "1": (101.3061224489796, 182.0, 240.3673469387755),
    "2": (112.15416666666667, 77.475, 49.62083333333333),
    "3": (56.38392857142857, 77.91964285714286, 201.66071428571428),
    "5": (131.54395604395606, 149.87912087912088, 229.03846153846155)
}