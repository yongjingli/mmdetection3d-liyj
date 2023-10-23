def debug_show_metric():
    metirc_infos = {'Gbld metric/line_instance': {'all': [0.85, 0.82, 0.834230999401556], 'road_boundary_line': [0.83, 1.0, 0.906608410704533], 'bushes_boundary_line': [0.75, 0.83, 0.7874762808349146], 'fence_boundary_line': [-1, -1, -1], 'stone_boundary_line': [0.5, 0.25, 0.33288948069241014], 'wall_boundary_line': [-1, -1, -1], 'water_boundary_line': [-1, -1, -1], 'snow_boundary_line': [-1, -1, -1], 'manhole_boundary_line': [1.0, 0.0, 0.0], 'others_boundary_line': [-1, -1, -1]}, 'Gbld metric/line_pixel': {'all': [0.69, 0.67, 0.68], 'road_boundary_line': [0.76, 0.79, 0.77], 'bushes_boundary_line': [0.64, 0.64, 0.64], 'fence_boundary_line': [-1, -1, -1], 'stone_boundary_line': [0.45, 0.16, 0.23], 'wall_boundary_line': [-1, -1, -1], 'water_boundary_line': [-1, -1, -1], 'snow_boundary_line': [-1, -1, -1], 'manhole_boundary_line': [0.49, 0.21, 0.29], 'others_boundary_line': [-1, -1, -1]}}

    for metirc_info in metirc_infos.keys():
        if "line_instance" in metirc_info:
            metirc_info = metirc_infos[metirc_info]
            print(metirc_info)


if __name__ == "__main__":
    print("Start")
    debug_show_metric()
    print("End")
