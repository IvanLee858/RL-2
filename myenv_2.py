"""
Environment for Robot Arm.
You can customize this script in a way you want.

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/


Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""
import numpy as np
import pyglet
# import time


pyglet.clock.set_fps_limit(10000)


class ArmEnv(object):
    action_bound = [-1, 1]
    action_dim = 2
    state_dim = 7  # 7 12 14
    dt = .1  # refresh rate
    arm1l = 100
    arm2l = 100
    viewer = None
    viewer_xy = (400, 400)
    get_point1 = False
    # point 2 
    get_point2 = False    
    mouse_in = np.array([False])
    point_l = 15
    grab_counter1 = 0
    grab_counter2 = 0

    def __init__(self, mode='easy'):
        # node1 (l, d_rad, x, y),
        # node2 (l, d_rad, x, y)
        self.mode = mode
        self.arm_info = np.zeros((2, 4))
        self.arm_info[0, 0] = self.arm1l
        self.arm_info[1, 0] = self.arm2l
        self.point1_info = np.array([250, 303])
        # point 2 
        self.point2_info = np.array([220, 273])        
        self.point1_info_init = self.point1_info.copy()
        self.point2_info_init = self.point2_info.copy()
        self.center_coord = np.array(self.viewer_xy)/2

    def step(self, action):
        # action = (node1 angular v, node2 angular v)
        action = np.clip(action, *self.action_bound)
        self.arm_info[:, 1] += action * self.dt
        self.arm_info[:, 1] %= np.pi * 2

        arm1rad = self.arm_info[0, 1]
        arm2rad = self.arm_info[1, 1]
        arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
        self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
        self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)
        
        s, arm2_distance1, arm2_distance2 = self._get_state()
        r = self._r_func(arm2_distance1, arm2_distance2)
        # if self.get_point1:
        #     print("in step", self.get_point1)

        # if self.get_point1:
        #     self.point1_info = self.arm_info[1, 2:4]

        return s, r, (self.get_point1 and self.get_point2), self.get_point1
        # return s, r,  self.get_point1


    def reset(self):
        self.get_point1 = False
        self.get_point2 = False
        self.grab_counter1 = 0
        self.grab_counter2 = 0

        if self.mode == 'hard':
            pxy1 = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 300)
            pxy2 = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 300)

            # To make box1 and box2 larger distance: 5
            while ((np.abs(pxy2[0] - pxy1[0]) <= 2*self.point_l+5) or (np.abs(pxy2[0] - pxy1[0]) <= 2*self.point_l+5)):
                pxy2 = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 300)
            self.point1_info[:] = pxy1
            self.point2_info[:] = pxy2

        else:
            arm1rad, arm2rad = np.random.rand(2) * np.pi * 2
            self.arm_info[0, 1] = arm1rad
            self.arm_info[1, 1] = arm2rad
            arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
            arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
            self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
            self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)

            self.point1_info[:] = self.point1_info_init
            self.point2_info[:] = self.point2_info_init
        return self._get_state()[0]

    def render(self,x):
        if self.viewer is None:
            # if self.get_point1:
            #     self.point1_info = self.arm_info[1, 2:4]  # to solve
            # if self.get_point1:
            #     print("in render", self.get_point1)
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point1_info, self.point2_info, self.point_l, self.mouse_in)
        # print("xxxxxxxxxxxx",x)
        self.viewer.render(x)

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self):
        # return the distance (dx, dy) between arm finger point with blue point
        arm_end = self.arm_info[:, 2:4]
        t_arms1 = np.ravel(arm_end - self.point1_info)
        t_arms2 = np.ravel(arm_end - self.point2_info)
        t_12 = (self.point1_info - self.point2_info)/200


        center_dis = (self.center_coord - self.point1_info)/200
        in_point1 = 1 if self.grab_counter1 > 0 else 0
        in_point2 = 1 if self.grab_counter2 > 0 else 0
        # print(t_arms1)
        # print(np.hstack([in_point1, in_point2, t_arms1/200, t_arms2/200, center_dis,
        #                   # arm1_distance_p, arm1_distance_b,
        #                   ]))

        # return np.hstack([in_point1, in_point2, t_arms1/200, center_dis, t_12, t_arms2/200,
        #                   # arm1_distance_p, arm1_distance_b,
        #                   ]), t_arms1[-2:], t_arms2[-2:]

        return np.hstack([in_point1, t_arms1/200, center_dis,
                          # arm1_distance_p, arm1_distance_b,
                          ]), t_arms1[-2:], t_arms2[-2:]

    def _r_func(self, distance1, distance2):
        t = 50
        abs_distance1= np.sqrt(np.sum(np.square(distance1)))
        abs_distance2= np.sqrt(np.sum(np.square(distance2)))
        r1 = -abs_distance1/200
        # r2 = -abs_distance2/200
        r2 = 0

        if abs_distance1 < self.point_l and (not self.get_point1):
            r1 += 1.
            self.grab_counter1 += 1
            if self.grab_counter1 > t:
                r1 += 10.
                self.get_point1 = True
                print("get point 1")
                if abs_distance2 < 2*self.point_l and (not self.get_point2): # to do: let it includes x and y
                    r2 += 1.
                    self.grab_counter2 += 1
                    if self.grab_counter2 > t:
                        r2 += 10.
                        self.get_point2 = True
                        print("get point 2")
                elif abs_distance2 > 2*self.point_l:
                    self.grab_counter2 = 0
                    self.get_point2 = False
                    print("get 1, but not get 2")
        elif abs_distance1 > self.point_l:
            self.grab_counter1 = 0
            self.get_point1 = False
        r = r1 + r2
        return r


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, arm_info, point1_info, point2_info, point_l, mouse_in):
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.arm_info = arm_info
        self.point1_info = point1_info
        self.point2_info = point2_info

        self.mouse_in = mouse_in
        self.point_l = point_l

        self.center_coord = np.array((min(width, height)/2, ) * 2)
        self.batch = pyglet.graphics.Batch()

        arm1_box, arm2_box, point1_box, point2_box = [0]*8, [0]*8, [0]*8, [0]*8
        c1, c2, c3 = (249, 86, 86)*4, (86, 109, 249)*4, (249, 39, 65)*4
        self.point1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point1_box), ('c3B', c2))
        self.point2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point2_box), ('c3B', c3))
        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm1_box), ('c3B', c1))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm2_box), ('c3B', c1))

    def render(self,x):
        pyglet.clock.tick()
        self._update_arm(x)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def _update_arm(self,x):
        point_l = self.point_l
        # if ArmEnv.get_point1:
        #     self.point1_info = ArmEnv.arm_info[0, 2:4]  # to solve
        
        # self.point1_info = [np.random.rand(),np.random.rand()
        # if self.get_point1:
        # print("22222in viwer render update arm", x)       
        if x:
            info1 = self.arm_info[1, 2:4]  # to solve
        else:
            info1 = self.point1_info



        point1_box = (info1[0] - point_l, info1[1] - point_l,
                     info1[0] + point_l, info1[1] - point_l,
                     info1[0] + point_l, info1[1] + point_l,
                     info1[0] - point_l, info1[1] + point_l)
        point2_box = (self.point2_info[0] - point_l, self.point2_info[1] - point_l,
                     self.point2_info[0] + point_l, self.point2_info[1] - point_l,
                     self.point2_info[0] + point_l, self.point2_info[1] + point_l,
                     self.point2_info[0] - point_l, self.point2_info[1] + point_l)
        self.point1.vertices = point1_box
        self.point2.vertices = point2_box

        arm1_coord = (*self.center_coord, *(self.arm_info[0, 2:4]))  # (x0, y0, x1, y1)
        arm2_coord = (*(self.arm_info[0, 2:4]), *(self.arm_info[1, 2:4]))  # (x1, y1, x2, y2)
        arm1_thick_rad = np.pi / 2 - self.arm_info[0, 1]
        x01, y01 = arm1_coord[0] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] + np.sin(
            arm1_thick_rad) * self.bar_thc
        x02, y02 = arm1_coord[0] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x11, y11 = arm1_coord[2] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x12, y12 = arm1_coord[2] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] + np.sin(
            arm1_thick_rad) * self.bar_thc
        arm1_box = (x01, y01, x02, y02, x11, y11, x12, y12)
        arm2_thick_rad = np.pi / 2 - self.arm_info[1, 1]
        x11_, y11_ = arm2_coord[0] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] - np.sin(
            arm2_thick_rad) * self.bar_thc
        x12_, y12_ = arm2_coord[0] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x21, y21 = arm2_coord[2] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x22, y22 = arm2_coord[2] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] - np.sin(
            arm2_thick_rad) * self.bar_thc
        arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)
        self.arm1.vertices = arm1_box
        self.arm2.vertices = arm2_box



    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            self.arm_info[0, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.DOWN:
            self.arm_info[0, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.LEFT:
            self.arm_info[1, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.RIGHT:
            self.arm_info[1, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.Q:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.A:
            pyglet.clock.set_fps_limit(30)

    # def on_mouse_motion(self, x, y, dx, dy):
    #     self.point_info[:] = [x, y]

    # def on_mouse_enter(self, x, y):
    #     self.mouse_in[0] = True

    # def on_mouse_leave(self, x, y):
    #     self.mouse_in[0] = False





# if __name__ == '__main__':
#     env = ArmEnv()
#     env.set_fps(30)
#     time.sleep(10)
#     s = env.reset()
#     time.sleep(10)
#     env.step([30/180*2*np.pi,6/180*2*np.pi])
#     env.render()
#     time.sleep(10)

