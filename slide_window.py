# coding: utf-8


def slide_window(img_width, img_height, width_min=10, height_min=10,
                 width_max=None, height_max=None, width_inc=10, height_inc=10,
                 x_step=3, y_step=3, ratio_min=None, ratio_max=None):
    """
    Produce slide_windows with given conditions.

    Parameters
    ----------
    img_width : int
        Width of the image.
    img_height : int
        Height of the image.
    width_min : int, optional (default=10)
        Minimum width of slide window.
    height_min : int, optional (default=10)
        Minimum height of slide window.
    width_max : int, optional
        Maximum width of slide window.
    height_max : int, optional
        Maximum height of slide window.
    width_inc : int, optional (default=3)
        Slide window's width will increase by width_inc.
    height_inc : int, optional (default=3)
        Slide window's height will increase by height_inc.
    x_step : int, optional (default=3)
        Slide window's x-coordinate will increase by x_step.
    y_step : int, optional (default=3)
        Slide window's y-coordinate will increase by y_step.
    ratio_min : float, optional
        Minimum value of height / width, it will decide slide window's
        minimum height.
        If None is given, window's minimum height will decide by height_min.
    ratio_max : float, optional
        Maximum value of height / width, it will decide slide window's
        maximum height.
        If None if given, window's maximum height will decide by height_max.

    Returns
    -------
    Yield tuples contains (y_start, y_end, x_start, x_end).
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
                    yield (y_pos, y_pos + h, x_pos, x_pos + w)
