import argparse
import copy
import gc
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow
from cv2.ximgproc import createGuidedFilter
from rich.traceback import install
from tqdm import tqdm

from src.utils import get_entries

tensorflow.compat.v1.disable_v2_behavior()
tf = tensorflow.compat.v1


def to_gray(x):
    R = x[:, :, :, 0:1]
    G = x[:, :, :, 1:2]
    B = x[:, :, :, 2:3]
    return 0.30 * R + 0.59 * G + 0.11 * B


def vgg2rgb(x):
    return (x + [103.939, 116.779, 123.68])[:, :, :, ::-1]


def nts(x):
    return (x + [103.939, 116.779, 123.68])[:, :, :, ::-1] / 255.0


def np_expand_image(x):
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), "symmetric")
    r = []
    r.append(p[:-2, 1:-1, :])
    r.append(p[1:-1, :-2, :])
    r.append(p[1:-1, 1:-1, :])
    r.append(p[1:-1, 2:, :])
    r.append(p[2:, 1:-1, :])
    return np.stack(r, axis=2)


def build_sketch_sparse(x, abs):
    x = x[:, :, None].astype(np.float32)
    expanded = np_expand_image(x)
    distance = x[:, :, None] - expanded
    if abs:
        distance = np.abs(distance)
    weight = 8 - distance
    weight[weight < 0] = 0.0
    weight /= np.sum(weight, axis=2, keepdims=True)
    return weight


def build_repeat_mulsep(x, m, i):
    a = m[:, :, 0]
    b = m[:, :, 1]
    c = m[:, :, 2]
    d = m[:, :, 3]
    e = m[:, :, 4]
    y = x
    for _ in range(i):
        p = tf.pad(y, [[1, 1], [1, 1], [0, 0]], "SYMMETRIC")
        y = (
            p[:-2, 1:-1, :] * a
            + p[1:-1, :-2, :] * b
            + y * c
            + p[1:-1, 2:, :] * d
            + p[2:, 1:-1, :] * e
        )
    return y


def make_graph():
    tail = tf.keras.models.load_model(f"{checkpoints_path}/tail.net")
    reader = tf.keras.models.load_model(f"{checkpoints_path}/reader.net")
    head = tf.keras.models.load_model(f"{checkpoints_path}/head.net")
    neck = tf.keras.models.load_model(f"{checkpoints_path}/neck.net")

    tail_op = tail(ip3)
    features = reader(ip3 / 255.0)
    logging.info("Loaded some basic models.")

    feed = [
        1 - ip1 / 255.0,
        (ip4[:, :, :, 0:3] / 127.5 - 1) * ip4[:, :, :, 3:4] / 255.0,
    ]
    for _ in range(len(features)):
        feed.append(tf.reduce_mean(features[_], axis=[1, 2]))
    nil0, nil1, head_temp = head(feed)
    feed[0] = tf.clip_by_value(
        1
        - tf.image.resize_bilinear(
            to_gray(vgg2rgb(head_temp) / 255.0), tf.shape(ip1)[1:3]
        ),
        0.0,
        1.0,
    )
    nil4, nil5, head_temp = neck(feed)
    head_op = vgg2rgb(head_temp)

    logging.info("Loaded - Style2Paints Deep Learning Engine V4.6 - GPU")

    session.run(tf.global_variables_initializer())

    tail.load_weights(f"{checkpoints_path}/tail.net")
    head.load_weights(f"{checkpoints_path}/head.net")
    neck.load_weights(f"{checkpoints_path}/neck.net")
    reader.load_weights(f"{checkpoints_path}/reader.net")

    logging.info("Deep learning modules are ready.")

    return tail_op, head_op


def go_tail(x):
    def srange(l, s):
        result = []
        iters = int(float(l) / float(s))
        for i in range(iters):
            result.append([i * s, (i + 1) * s])
        result[len(result) - 1][1] = l
        return result

    H, W, C = x.shape
    padded_img = (
        np.pad(x, ((20, 20), (20, 20), (0, 0)), "symmetric").astype(np.float32) / 255.0
    )
    lines = []
    for hs, he in srange(H, 64):
        items = []
        for ws, we in srange(W, 64):
            items.append(padded_img[hs : he + 40, ws : we + 40, :])
        lines.append(items)
    iex = 0
    result_all_lines = []
    for line in lines:
        result_one_line = []
        for item in line:
            ots = session.run(tail_op_g, feed_dict={ip3: item[None, :, :, :]})[0]
            result_one_line.append(ots[41:-41, 41:-41, :])
            iex += 1
        result_one_line = np.concatenate(result_one_line, axis=1)
        result_all_lines.append(result_one_line)
    result_all_lines = np.concatenate(result_all_lines, axis=0)
    return (result_all_lines * 255.0).clip(0, 255).astype(np.uint8)


def go_head(sketch, global_hint, local_hint):
    return (
        session.run(
            head_op_g,
            feed_dict={
                ip1: sketch[None, :, :, None],
                ip3: global_hint[None, :, :, :],
                ip4: local_hint[None, :, :, :],
            },
        )[0]
        .clip(0, 255)
        .astype(np.uint8)
    )


def k_resize(x, k):
    if x.shape[0] < x.shape[1]:
        s0 = k
        s1 = int(x.shape[1] * (k / x.shape[0]))
        s1 = s1 - s1 % 64
        _s0 = 16 * s0
        _s1 = int(x.shape[1] * (_s0 / x.shape[0]))
        _s1 = (_s1 + 32) - (_s1 + 32) % 64
    else:
        s1 = k
        s0 = int(x.shape[0] * (k / x.shape[1]))
        s0 = s0 - s0 % 64
        _s1 = 16 * s1
        _s0 = int(x.shape[0] * (_s1 / x.shape[1]))
        _s0 = (_s0 + 32) - (_s0 + 32) % 64
    new_min = min(_s1, _s0)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
    return y


def d_resize(x, d, fac=1.0):
    new_min = min(int(d[1] * fac), int(d[0] * fac))
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
    return y


def min_resize(x, m):
    if x.shape[0] < x.shape[1]:
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
    new_max = min(s1, s0)
    raw_max = min(x.shape[0], x.shape[1])
    if new_max < raw_max:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (s1, s0), interpolation=interpolation)
    return y


def cli_norm(sketch):
    light = np.max(min_resize(sketch, 64), axis=(0, 1), keepdims=True)
    intensity = (light - sketch.astype(np.float32)).clip(0, 255)
    line_intensities = np.sort(intensity[intensity > 16])[::-1]

    if len(line_intensities) != 0:
        line_quantity = float(line_intensities.shape[0])
        intensity /= line_intensities[int(line_quantity * 0.1)]
        intensity *= 0.9

    return (255.0 - intensity * 255.0).clip(0, 255).astype(np.uint8)


def from_png_to_jpg(map):
    if map.shape[2] == 3:
        return map
    color = map[:, :, 0:3].astype(np.float) / 255.0
    alpha = map[:, :, 3:4].astype(np.float) / 255.0
    reversed_color = 1 - color
    final_color = (255.0 - reversed_color * alpha * 255.0).clip(0, 255).astype(np.uint8)
    return final_color


def s_enhance(x, k=2.0):
    p = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float)
    p[:, :, 1] *= k
    p = p.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(p, cv2.COLOR_HSV2RGB).clip(0, 255)


def ini_hint(x):
    r = np.zeros(shape=(x.shape[0], x.shape[1], 4), dtype=np.uint8)
    return r


def opreate_normal_hint(gird, points, length):
    h = gird.shape[0]
    w = gird.shape[1]
    for point in points:
        x, y, r, g, b = point
        x = int(x * w)
        y = int(y * h)
        l_ = max(0, x - length)
        b_ = max(0, y - length)
        r_ = min(w, x + length + 1)
        t_ = min(h, y + length + 1)
        gird[b_:t_, l_:r_, 2] = r
        gird[b_:t_, l_:r_, 1] = g
        gird[b_:t_, l_:r_, 0] = b
        gird[b_:t_, l_:r_, 3] = 255.0
    return gird


def get_hdr(x):
    def get_hdr_g(x):
        img = x.astype(np.float32)
        mean = np.mean(img)
        h_mean = mean.copy()
        l_mean = mean.copy()
        for i in range(2):
            h_mean = np.mean(img[img >= h_mean])
            l_mean = np.mean(img[img <= l_mean])
        for i in range(2):
            l_mean = np.mean(img[img <= l_mean])
        return l_mean, mean, h_mean

    l_mean = np.zeros(shape=(1, 1, 3), dtype=np.float32)
    mean = np.zeros(shape=(1, 1, 3), dtype=np.float32)
    h_mean = np.zeros(shape=(1, 1, 3), dtype=np.float32)
    for c in range(3):
        l, m, h = get_hdr_g(x[:, :, c])
        l_mean[:, :, c] = l
        mean[:, :, c] = m
        h_mean[:, :, c] = h
    return l_mean, mean, h_mean


def f2(x1, x2, x3, y1, y2, y3, x):
    A = y1 * ((x - x2) * (x - x3)) / ((x1 - x2) * (x1 - x3))
    B = y2 * ((x - x1) * (x - x3)) / ((x2 - x1) * (x2 - x3))
    C = y3 * ((x - x1) * (x - x2)) / ((x3 - x1) * (x3 - x2))
    return A + B + C


def refine_image(image, sketch, origin):
    sketch = sketch.astype(np.float32)
    sparse_matrix = build_sketch_sparse(sketch, True)
    guided_matrix = createGuidedFilter(sketch.clip(0, 255).astype(np.uint8), 1, 0.01)
    HDRL, HDRM, HDRH = get_hdr(image)

    def go_guide(x):
        y = x + (x - cv2.GaussianBlur(x, (0, 0), 1)) * 2.0
        for _ in range(4):
            y = guided_matrix.filter(y)
        return y

    def go_refine_sparse(x):
        return session.run(tf_sparse_op_H, feed_dict={ipsp3: x, ipsp9: sparse_matrix})

    def go_hdr(x):
        xl, xm, xh = get_hdr(x)
        y = f2(xl, xm, xh, HDRL, HDRM, HDRH, x)
        return y.clip(0, 255)

    def go_blend(BGR, X, m):
        BGR = BGR.clip(0, 255).astype(np.uint8)
        X = X.clip(0, 255).astype(np.uint8)
        YUV = cv2.cvtColor(BGR, cv2.COLOR_BGR2YUV)
        s_l = YUV[:, :, 0].astype(np.float32)
        t_l = X.astype(np.float32)
        r_l = (s_l * t_l / 255.0) if m else np.minimum(s_l, t_l)
        YUV[:, :, 0] = r_l.clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR)

    smoothed = d_resize(image, sketch.shape)
    sparse_smoothed = go_refine_sparse(smoothed)
    smoothed = go_guide(sparse_smoothed)
    smoothed = go_hdr(smoothed)
    blended_smoothed = go_blend(smoothed, origin, False)

    return smoothed, blended_smoothed


def colorize(sketch, face):
    def transform(sketch):
        origin = from_png_to_jpg(sketch)
        sketch = min_resize(origin, 512)
        sketch = np.min(sketch, axis=2)
        sketch = cli_norm(sketch)
        sketch = np.tile(sketch[:, :, None], [1, 1, 3])
        sketch = go_tail(sketch)
        sketch = np.mean(sketch, axis=2)
        return copy.deepcopy(origin), copy.deepcopy(sketch)

    origin, sketch = transform(sketch)
    points = []

    if origin.ndim == 3:
        origin = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
    origin = d_resize(origin, sketch.shape).astype(np.float32)
    low_origin = cv2.GaussianBlur(origin, (0, 0), 3.0)
    high_origin = origin - low_origin
    low_origin = (low_origin / np.median(low_origin) * 255.0).clip(0, 255)
    origin = (low_origin + high_origin).clip(0, 255).astype(np.uint8)

    face = from_png_to_jpg(face)
    face = s_enhance(face, 2.0)

    sketch_1024 = k_resize(sketch, 64)
    hints_1024 = opreate_normal_hint(ini_hint(sketch_1024), points, length=2)
    careless = go_head(sketch_1024, k_resize(face, 14), hints_1024)

    return refine_image(careless, sketch, origin)


def setup_globals(checkpoints_path_args):
    global checkpoints_path, session, ip1, ip3, ip4, ipsp9, ipsp3, tf_sparse_op_H, tail_op_g, head_op_g
    checkpoints_path = checkpoints_path_args

    session = tf.Session()
    tf.keras.backend.set_session(session)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    ip1 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 1))
    ip3 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
    ip4 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 4))
    ipsp9 = tf.placeholder(dtype=tf.float32, shape=(None, None, 5, 1))
    ipsp3 = tf.placeholder(dtype=tf.float32, shape=(None, None, 3))

    tf_sparse_op_H = build_repeat_mulsep(ipsp3, ipsp9, 64)
    tail_op_g, head_op_g = make_graph()

    logging.info("Deep learning functions are ready.")
    logging.info("Tricks loaded.")
    logging.info("Fundamental Methods loaded.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints_path",
        type=str,
        required=True,
        help="Path to the model checkpoints directory",
    )
    parser.add_argument(
        "--colored_path",
        type=str,
        required=True,
        help="Path to the colored input directory",
    )
    parser.add_argument(
        "--style2paints_path",
        type=str,
        required=True,
        help="Path to the style2paints output directory",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Setup globals
    setup_globals(args.checkpoints_path)

    # Create output directory
    logging.info("Creating output directory")
    os.makedirs(args.style2paints_path, exist_ok=True)

    # Conversion
    logging.info("Start conversion")
    colored_files = get_entries(f"{args.colored_path}/*.png")
    with tqdm(colored_files, unit="sample") as tbatch:
        for sample, colored_file in enumerate(tbatch):
            colored = cv2.imread(colored_file, cv2.IMREAD_UNCHANGED)
            smoothed, blended_smoothed = colorize(
                copy.deepcopy(colored), copy.deepcopy(colored)
            )
            path = f"{args.style2paints_path}/{sample:06}.png"
            cv2.imwrite(path, blended_smoothed)

    # Clean up
    gc.collect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    install(show_locals=False)
    args = parse_args()
    main(args)
