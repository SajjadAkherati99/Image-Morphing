import dlib
import cv2
from imutils import face_utils
import numpy as np
import sys
import os
from scipy.spatial import Delaunay
import moviepy.video.io.ImageSequenceClip
import time


def generate_color_vec(length):
    color_vec = np.zeros([length, 3]).astype('int')
    for i in range(length):
        r, g, b = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
        color_vec[i, :] = [b, g, r]

    return color_vec


def resize_images(img1, img2, size=None):
    if size is None:
        size = [400, 400]
    img1 = cv2.resize(img1, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(img2, (size[0], size[1]), interpolation=cv2.INTER_AREA)
    return img1, img2


def find_points(img, num_of_points = 24):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    rects = detector(img, 1)
    points = []
    for i, rect in enumerate(rects):
        points = predictor(img, rect)
        points = face_utils.shape_to_np(points)
    point = np.zeros([num_of_points-4, 2]).astype('int')

    for i in range(num_of_points-4):
        ind = np.round(i*67/(num_of_points-5)).astype('int')
        point[i, :] = points[ind, :]

    points = np.append(point, [[0, 0]], axis=0)
    points = np.append(points, [[0, img.shape[0]]], axis=0)
    points = np.append(points, [[img.shape[1], 0]], axis=0)
    points = np.append(points, [[img.shape[1], img.shape[0]]], axis=0)
    return points


def find_triangle_vertexes(points):
    triangle_vertexes = Delaunay(points)
    return triangle_vertexes.simplices


def draw_triangles(img, verts, vert_cords):
    for i in range(verts.shape[0]):
        p1, p2, p3 = tuple(vert_cords[verts[i, 0], :]), \
                     tuple(vert_cords[verts[i, 1], :]), tuple(vert_cords[verts[i, 2], :])
        cv2.line(img, p1, p2, (0, 0, 255), 1)
        cv2.line(img, p2, p3, (0, 0, 255), 1)
        cv2.line(img, p1, p3, (0, 0, 255), 1)
    return img


def separate_triangles(vert, vert_cords, img_shape):
    labels = np.zeros([img_shape[0], img_shape[1]]).astype('int') - 1
    x_cords, y_cords = np.zeros([img_shape[0], img_shape[1]]).astype('int'), \
                       np.zeros([img_shape[0], img_shape[1]]).astype('int')
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            x_cords[i, j], y_cords[i, j] = i, j
    for i in range(vert.shape[0]):
        p1, p2, p3 = vert_cords[vert[i, 0], :], \
                     vert_cords[vert[i, 1], :], vert_cords[vert[i, 2], :]
        labels[(((((p2[1] - p1[1]) * (y_cords - p1[0]) - (p2[0] - p1[0]) * (x_cords - p1[1])) <= 0) &
                 (((p3[1] - p2[1]) * (y_cords - p2[0]) - (p3[0] - p2[0]) * (x_cords - p2[1])) <= 0) &
                 (((p1[1] - p3[1]) * (y_cords - p3[0]) - (p1[0] - p3[0]) * (x_cords - p3[1])) <= 0) &
                 (labels == -1)) |
                ((((p2[1] - p1[1]) * (y_cords - p1[0]) - (p2[0] - p1[0]) * (x_cords - p1[1])) >= 0) &
                 (((p3[1] - p2[1]) * (y_cords - p2[0]) - (p3[0] - p2[0]) * (x_cords - p2[1])) >= 0) &
                 (((p1[1] - p3[1]) * (y_cords - p3[0]) - (p1[0] - p3[0]) * (x_cords - p3[1])) >= 0) &
                 (labels == -1)))] = i

    return labels


def colorize_triangles(labels, color_vec):
    img = np.zeros([labels.shape[0], labels.shape[1], 3]).astype('int')
    for i in range(len(np.unique(labels))):
        x, y = np.where(labels == i)
        img[x, y, :] = color_vec[i, :]
    return img


def warping(img, label, src_vert, src_vert_cords, dst_vert, dst_vert_cords):
    img_out = np.zeros(img.shape).astype('int')
    for i in range(np.max(label)+1):
        # print(' ', i)
        img1 = np.copy(img)
        img1[label!=i] = 0
        rows, cols, ch = img1.shape
        pts1 = np.float32([[src_vert_cords[src_vert[i, 0], :],
                            src_vert_cords[src_vert[i, 1], :],
                            src_vert_cords[src_vert[i, 2], :]]])
        pts2 = np.float32([[dst_vert_cords[dst_vert[i, 0], :],
                            dst_vert_cords[dst_vert[i, 1], :],
                            dst_vert_cords[dst_vert[i, 2], :]]])
        M = cv2.getAffineTransform(pts1, pts2)
        warped = cv2.warpAffine(img1, M, (cols, rows))
        img_out += warped
    return img_out


def morphing(img1, img2, label1, label2, vert1, vert2, vert_cord1, vert_cord2, dirname, num_of_frames=100):
    uv = vert_cord2-vert_cord1
    for i in range(num_of_frames+1):
        # print(i)
        alpha = i/num_of_frames
        vert_dst_cord = vert_cord1 + np.round(alpha*uv).astype('int')
        warped1 = (warping(img1, label1, vert1, vert_cord1, vert1, vert_dst_cord))
        warped2 = (warping(img2, label2, vert2, vert_cord2, vert2, vert_dst_cord))
        img3 = (1-alpha) * warped1 + alpha * warped2
        cv2.imwrite(os.path.join(dirname, '{i}.jpg'.format(i = i)), img3)
    return


def make_pic_folder():
    file = sys.argv[0]
    dirname = os.path.dirname(file)
    dirname = dirname + '/' + 'pic'
    if os.path.exists(dirname) == 0:
        os.mkdir(dirname)
    return dirname


def make_gif_file(dirname, fps=1, num_of_frames=100):
    image_folder = dirname
    image_files = []
    for i in range(num_of_frames+1):
        image_files.append(image_folder + '/' + str(i) + ".jpg")
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile('res2.mp4')
    return


start_time = time.time()
img1 = cv2.imread('johnnydepp1.jpg')
img2 = cv2.imread('johnnydepp2.jpg')

img1, img2 = resize_images(img1, img2, size=[300, 300])
cv2.imwrite('sajjad1_resized.jpg', img1)
cv2.imwrite('johnnydepp_resized.jpg', img2)

points1, points2 = find_points(img1, num_of_points=20), find_points(img2, num_of_points=20)
triangle_vertexes1, triangle_vertexes2 = (find_triangle_vertexes(points1)), \
                                         (find_triangle_vertexes(points2))

img3 = draw_triangles(np.copy(img1), triangle_vertexes1, points1)
img4 = draw_triangles(np.copy(img2), triangle_vertexes2, points2)
cv2.imwrite('sajjad1_triangles.jpg', img3)
cv2.imwrite('johnnydepp_triangles.jpg', img4)

labels1, labels2 = separate_triangles(triangle_vertexes1, points1, img1.shape),\
                   separate_triangles(triangle_vertexes2, points2, img2.shape)
color_vec = generate_color_vec(max(len(np.unique(labels1)), len(np.unique(labels2))))
img5, img6 = colorize_triangles(labels1, color_vec), colorize_triangles(labels2, color_vec)
cv2.imwrite('triangles1.jpg', img5)
cv2.imwrite('triangles2.jpg', img6)

execution_time0 = time.time() - start_time
print('execution time before morphing = ', execution_time0)
#
dirname = make_pic_folder()
morphing(img1, img2, labels1, labels2, triangle_vertexes1, triangle_vertexes2, points1, points2,
         dirname, num_of_frames=100)
make_gif_file(dirname, fps=30, num_of_frames=100)

execution_time = time.time() - start_time
print('execution time of all = ', execution_time)
