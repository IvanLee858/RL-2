import multiprocessing
import threading
import tensorflow as tf
import numpy as np
from arm_env import ArmEnv


# np.random.seed(1)
# tf.set_random_seed(1)

episodes = 2000
max_step= 300
update_iter = 5
N_WORKERS = multiprocessing.cpu_count()
alphaA = 1e-4  # learning rate for actor
alphaC = 2e-4  # learning rate for critic
gamma = 0.9  # reward discount
MODE = ['easy', 'hard']
n_model = 1
GLOBAL_NET_SCOPE = 'Global_Net'
ENTROPY_BETA = 0.01
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0


env = ArmEnv(mode=MODE[n_model])
state_space = env.state_dim
action_space  = env.action_dim
action_bound = env.action_bound
del env


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.compat.v1.variable_scope(scope):
                self.s = tf.compat.v1.placeholder(tf.float32, [None, state_space ], 'S')
                self._build_net()
                self.a_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:   # local net, calculate losses
            with tf.compat.v1.variable_scope(scope):
                self.s = tf.compat.v1.placeholder(tf.float32, [None, state_space ], 'S')
                self.a_his = tf.compat.v1.placeholder(tf.float32, [None, action_space ], 'A')
                self.v_target = tf.compat.v1.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v = self._build_net()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.compat.v1.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(input_tensor=tf.square(td))

                with tf.compat.v1.name_scope('wrap_a_out'):
                    self.test = sigma[0]
                    mu, sigma = mu * action_bound[1], sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(mu, sigma)

                with tf.compat.v1.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(input_tensor=-self.exp_v)

                with tf.compat.v1.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *action_bound)
                with tf.compat.v1.name_scope('local_grad'):
                    self.a_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(ys=self.a_loss, xs=self.a_params)
                    self.c_grads = tf.gradients(ys=self.c_loss, xs=self.c_params)

            with tf.compat.v1.name_scope('sync'):
                with tf.compat.v1.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.compat.v1.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        with tf.compat.v1.variable_scope('actor'):
            l_a = tf.compat.v1.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a = tf.compat.v1.layers.dense(l_a, 300, tf.nn.relu6, kernel_initializer=w_init, name='la2')
            mu = tf.compat.v1.layers.dense(l_a, action_space , tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.compat.v1.layers.dense(l_a, action_space , tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.compat.v1.variable_scope('critic'):
            l_c = tf.compat.v1.layers.dense(self.s, 400, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_c = tf.compat.v1.layers.dense(l_c, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc2')
            v = tf.compat.v1.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        return mu, sigma, v

    def update_global(self, feed_dict):  # run by a local
        _, _, t = SESS.run([self.update_a_op, self.update_c_op, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})[0]


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = ArmEnv(mode=MODE[n_model])
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < episodes:
            s = self.env.reset()
            ep_r = 0
            for ep_t in range(max_step):
                if self.name == 'W_0':
                    self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done = self.env.step(a)
                if ep_t == max_step- 1: done = True
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % update_iter == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    test = self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                        '| Var:', test,

                          )
                    GLOBAL_EP += 1
                    break

if __name__ == "__main__":
    SESS = tf.compat.v1.Session ()

    with tf.device("/cpu:0"):
        OPT_A = tf.compat.v1.train.RMSPropOptimizer(alphaA, name='RMSPropA')
        OPT_C = tf.compat.v1.train.RMSPropOptimizer(alphaC, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.compat.v1.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
