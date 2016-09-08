# coding: utf-8
"""
slide_window
============
提供滑动窗口函数，供液位仪与窗口定位使用。
"""


def slide_window(img_width, img_height, width_min=10, height_min=10,
                 width_max=None, height_max=None, width_inc=10, height_inc=10,
                 x_step=3, y_step=3, ratio_min=None, ratio_max=None):
    """
    **滑动窗口**

    由给定的条件生成合适的滑动窗口的位置。

    Parameters
    ----------
    img_width : int
        图像的宽。
    img_height : int
        图像的高。
    width_min : int, optional (default=10)
        滑动窗口最小宽度。
    height_min : int, optional (default=10)
        滑动窗口最小高度。
    width_max : int, optional
        滑动窗口最大宽度。
    height_max : int, optional
        滑动窗口最大高度。
    width_inc : int, optional (default=3)
        滑动窗口宽度每次增加的值。
    height_inc : int, optional (default=3)
        滑动窗口高度每次增加的值。
    x_step : int, optional (default=3)
        滑动窗口在横向上每次移动的距离。
    y_step : int, optional (default=3)
        滑动窗口在竖向上每次移动的距离。
    ratio_min : float, optional
        滑动窗口高/宽的最小值。
    ratio_max : float, optional
        滑动窗口高/宽的最大值。

    Yields
    -------
    tuple
        滑动窗口位置。
    """
    if not width_max or width_max > img_width:  # find maximum window width
        width_max = img_width

    for w in range(width_min, width_max, width_inc):
        # find minimum window height
        h_min = height_min
        if ratio_min and w * ratio_min > h_min:
            h_min = int(w * ratio_min)

        # find maximum window height
        h_max = [img_height]
        if ratio_max:
            h_max.append(int(w * ratio_max))
        if height_max:
            h_max.append(height_max)
        h_max = min(h_max)

        for h in range(h_min, h_max, height_inc):
            for x_pos in range(0, img_width - w, x_step):
                for y_pos in range(0, img_height - h, y_step):
                    # produce every window
                    yield (y_pos, y_pos+h, x_pos, x_pos+w)
