video_table_data = {
    "hsv_color_lower": (73, 100, 0),
    "hsv_color_upper": (87, 255, 255)
}

balls_data = {
    "0":
        {"hsv_color_lower": (25, 30, 160),
         "hsv_color_upper": (60, 120, 255),
         "trajectory_color": (255, 255, 255)
         },
    # "2":
    #     {"hsv_color_lower": (99, 10, 50),
    #      "hsv_color_upper": (119, 255, 150),
    #      "trajectory_color": (112, 77, 49)
    #      },
    "3":
        {"hsv_color_lower": (170,230,120),
         "hsv_color_upper": (182,255,225),
         "trajectory_color": (56, 144, 249)
         },
    "4":
        {"hsv_color_lower": (160, 50, 100),
         "hsv_color_upper": (180, 200, 255),
         "trajectory_color": (254, 102, 255)
         },
    "5":
        {"hsv_color_lower": (0, 200, 0),
         "hsv_color_upper": (18, 255, 255),
         "trajectory_color": (37, 61, 231)
         },
    "6":
        {"hsv_color_lower": (64, 200, 30),
         "hsv_color_upper": (80, 255, 120),
         "trajectory_color": (55, 58, 13)
         },
    # "7":
    #     {"hsv_color_lower": (12, 130, 130),
    #      "hsv_color_upper": (25, 200, 200),
    #      "trajectory_color": (66, 129, 175)
    #      },
    "8":
        {"hsv_color_lower": (40, 10, 0),
         "hsv_color_upper": (100, 255, 40),
         "trajectory_color": (0, 0, 0)
         },
    "9":
        {"hsv_color_lower": (10, 150, 50),
         "hsv_color_upper": (28, 255, 250),
         "trajectory_color": (73, 178, 243)
         }
}

sim_ball_data = {
    "0":
        {"ball_color": (255, 255, 255), "ball_image": None},
    "3":
        {"ball_color": (56, 144, 249), "ball_image": None},
    # "2":
    #     {"ball_color": (112, 77, 49), "ball_image": None},
    "4":
        {"ball_color": (254, 102, 255), "ball_image": None},
    "5":
        {"ball_color": (37, 61, 231), "ball_image": None},
    "6":
        {"ball_color": (55, 58, 13), "ball_image": None},
    # "7":
    #     {"ball_color": (66, 129, 175), "ball_image": None},
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