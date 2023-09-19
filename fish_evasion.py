# python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:41:02 2019

@author: yusheng
"""
# ------------------------------------------------------------------------
# required pacckages: NumPy, SciPy, openAI gym
# written in the framwork of gym
# ------------------------------------------------------------------------
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy import integrate
from colorama import Fore, Back, Style

class FishEvasionEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, mode = 'first-order', dt = 0.1, bodylength = 5):
        # size and mass parameters
        class param:
            a = np.repeat(bodylength,3)
            b = np.array([1,1,1])
            c = np.array([5,5,5])
            rho = 1000
        
        # length non-dimensionalization 
        L_c = 2*np.sum(param.a)
        a = param.a/L_c
        b = param.b/L_c
        c = param.c/L_c

        self.a,self.b,self.c = a,b,c
        self.rho = param.rho
        # characteristic mass
        # M_c = 4/3*self.rho*np.sum(a*b*c)*np.pi
        
        self.m = a*b*c/np.sum(a*b*c)*0
        self.J = (1/5*(a**2+b**2) + a**2)*self.m
        self.__added_mass()
        self.dt = dt
        self.oldpos = None
        
        # range of observtion variables
        high = np.array([np.finfo(np.float32).max, np.pi, np.pi])
        # create the observation space and the action space
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low = -1., high = 1., shape = (2,), dtype = np.float32)

        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        terminal = False
        dt = self.dt
        self.oldpos = list(self.pos)
        # compute the fish nose position
        Xhead_old = self.pos[0] + np.cos(self.pos[2])*self.a[0] + np.cos(self.pos[2]+self.shape[1])*self.a[2]*2

        self.shapechange = action
        # impose the contraint on the shape (angles cannot exceed 120 degrees):
        constraint_angle = 2*np.pi/3
        self.shapechange = np.clip(self.shapechange, (-constraint_angle - self.shape)/dt, (constraint_angle - self.shape)/dt)
        
        # integrate the dynamics system
        options = {'rtol':1e-4,'atol':1e-8,'max_step': 1e-2}
        sol = integrate.solve_ivp(self.__firstorderdt, (0,dt), self.pos, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False,**options)

        self.shape = self.shape + self.shapechange*dt              # update the shape
        self.time = self.time + dt                                 # update the time
        
        position_new = sol.y[:,-1]
        self.pos = position_new                           # update the position

        dXhead = self.pos[0] + np.cos(self.pos[2])*self.a[0] + np.cos(self.pos[2]+self.shape[1])*self.a[2]*2 - Xhead_old
        terminal = self.__terminal()                      # termination condition

        reward = dXhead                        # compute the reward (displacement of fish nose in x direction)
        return self._get_obs(), reward, terminal, {}
    def __firstorderdt(self,t,var):                   # mechanics
            m, ma1, ma2, J, Ja = self.m, self.ma1, self.ma2, self.J, self.Ja
            Dalpha_1, Dalpha_2 = self.shapechange
            alpha_1, alpha_2 = np.clip(self.shape + self.shapechange*t,-2*np.pi/3, 2*np.pi/3)
            # ------------------------------------------------------------------------
            """Used only when manually prescribing the shape change:"""
            # alpha_1, alpha_2, Dalpha_1, Dalpha_2 = self.__prescribedAngle(t+self.time, position = None)
            # self.shape = np.array([alpha_1, alpha_2])
            # self.shapechange = np.array([Dalpha_1,Dalpha_2])
            # ------------------------------------------------------------------------
            self.pos = var
            a1 = self.a[1]
            a2 = self.a[2]
            beta_m = self.pos[2]
            # 2D transformation matrix
            Q = np.array([[np.cos(beta_m), -np.sin(beta_m)],
                [np.sin(beta_m), np.cos(beta_m)]])
            # matrices below are defined a little different than what are given in the paper,
            # but eventually V@M gives the locked mass matrix and V@eta gives the coupling matrix
            V = self.__V()
            M = self.__M()
            eta = np.zeros((9,2))
            eta[3,0] = a1*(m[1]+ma2[1])
            eta[7,0] = -(J[1]+Ja[1])
            eta[5,1] = a2*(m[2]+ma2[2])
            eta[8,1] = (J[2]+Ja[2])
            A = -np.linalg.solve(V@M, V@eta)
            # equations of motions:
            vel = np.block([[Q, np.zeros((2,1))],[0,0,1]])@A@np.array([[Dalpha_1],[Dalpha_2]]).reshape((-1,))
            self.vel = vel
            return vel 
    def __added_mass(self):                            # compute the added mass
        a = self.a*2/2
        b = self.b*2/2
        c = self.c*2/2
        abc = a*b*c
        factor = np.sum(abc)
        def delta(lam,a,b,c):
            return np.sqrt((a**2+lam)*(b**2+lam)*(c**2+lam))
        def integrand_alpha(lam,a,b,c):
            return 1/(a**2+lam)/delta(lam,a,b,c)
        def integrand_beta(lam,a,b,c):
            return 1/(b**2+lam)/delta(lam,a,b,c)

        alpha = np.zeros_like(a)
        beta = np.zeros_like(a)
        ma1 = np.zeros_like(a)
        ma2= np.zeros_like(a)
        Ja = np.zeros_like(a)
        for i in range(a.size):
            alpha[i], err = integrate.quad(lambda lam: integrand_alpha(lam,self.a[i],self.b[i],self.c[i]),0,np.inf)
            beta[i], err = integrate.quad(lambda lam: integrand_beta(lam,self.a[i],self.b[i],self.c[i]),0,np.inf)
        alpha *= abc
        beta *= abc
        ma1 = alpha/(2-alpha[i])*abc/factor
        ma2 = beta/(2-beta[i])*abc/factor
        Ja = 1/5*(a**2 - b**2)**2*(beta - alpha)/(2*(a**2 - b**2) + (a**2 + b**2)*(alpha - beta))*abc/factor
        
        Ja[1:] = Ja[1:] + a[1:]**2*ma2[1:]
        self.ma1 = ma1
        self.ma2 = ma2
        self.Ja = Ja
    def __initialConfig(self,mode, init_num):                    # get the initial configuration of the fish
        X, Y, theta = 0, 0, np.random.rand()*2*np.pi-np.pi
        Alpha_1, Alpha_2 = np.random.rand()*2*np.pi/3 - np.pi/3, np.random.rand()*2*np.pi/3 - np.pi/3
        
        # if manually prescribing the shape change:
        # Alpha_1, Alpha_2,_ ,_ = self.__prescribedAngle(0, position = None)
        return np.array([X, Y, theta]), np.array([Alpha_1, Alpha_2])
    def __M(self):
        m, ma1, ma2, J, Ja = self.m, self.ma1, self.ma2, self.J, self.Ja
        a = self.a
        alpha_1 = self.shape[0]
        alpha_2 = self.shape[1]

        matrixM = np.zeros((9,3))
        
        matrixM[0,:] = np.array([m[0]+ma1[0], 0, 0])
        matrixM[1,:] = np.array([0, m[0]+ma2[0], 0])
        matrixM[2,:] = np.array([(m[1]+ma1[1])*np.cos(alpha_1), -(m[1]+ma1[1])*np.sin(alpha_1), (m[1]+ma1[1])*a[0]*np.sin(alpha_1)])
        matrixM[3,:] = np.array([(m[1]+ma2[1])*np.sin(alpha_1), (m[1]+ma2[1])*np.cos(alpha_1), -(m[1]+ma2[1])*(a[0]*np.cos(alpha_1)+a[1])])
        matrixM[4,:] = np.array([(m[2]+ma1[2])*np.cos(alpha_2), (m[2]+ma1[2])*np.sin(alpha_2), (m[2]+ma1[2])*a[0]*np.sin(alpha_2)])
        matrixM[5,:] = np.array([-(m[2]+ma2[2])*np.sin(alpha_2), (m[2]+ma2[2])*np.cos(alpha_2), (m[2]+ma2[2])*(a[0]*np.cos(alpha_2)+a[2])])
        matrixM[6,:] = np.array([0, 0, J[0]+Ja[0]])
        matrixM[7,:] = np.array([-a[1]*(m[1]+ma2[1])*np.sin(alpha_1), -a[1]*(m[1]+ma2[1])*np.cos(alpha_1), (J[1]+Ja[1])+(m[1]+ma2[1])*a[0]*a[1]*np.cos(alpha_1)])
        matrixM[8,:] = np.array([-a[2]*(m[2]+ma2[2])*np.sin(alpha_2), a[2]*(m[2]+ma2[2])*np.cos(alpha_2), (J[2]+Ja[2])+(m[2]+ma2[2])*a[0]*a[2]*np.cos(alpha_2)])
        return matrixM
    def __V(self):
        a_c = self.a[0]
        alpha_1, alpha_2 = self.shape
        matrixV = np.zeros((3,9))
        R1 = np.array([[np.cos(alpha_1), -np.sin(alpha_1)], [np.sin(alpha_1), np.cos(alpha_1)]])
        R2 = np.array([[np.cos(alpha_2), np.sin(alpha_2)], [-np.sin(alpha_2), np.cos(alpha_2)]])
        
        matrixV[0:2,0:6] = np.concatenate((np.identity(2),R1.T,R2.T), axis=1)
        matrixV[2,0:6] = [0, 0, a_c*np.sin(alpha_1), -a_c*np.cos(alpha_1), a_c*np.sin(alpha_2), a_c*np.cos(alpha_2)]
        matrixV[2,6:9] = [1,1,1]
        return matrixV
    def __prescribedAngle(self,time, position = None):
        # Used when manually precribing the shape change (disregarding the RL policy, mainly for test)
        """Experimental fitting path"""
#        a0 =       17.94
#        a1 =      -1.059
#        b1 =       23.23
#        a2 =       -29.4
#        b2 =       30.49
#        a3 =       14.93
#        b3 =      -31.43
#        w =      0.7886
#        
#        alpha_2 = a0 + a1*np.cos(time*w) + b1*np.sin(time*w) + a2*np.cos(2*time*w) + b2*np.sin(2*time*w) + a3*np.cos(3*time*w) + b3*np.sin(3*time*w)
#        Dalpha_2 = -w*a1*np.sin(time*w) + w*b1*np.cos(time*w) - 2*w*a2*np.sin(2*time*w) + 2*w*b2*np.cos(2*time*w) - 3*w*a3*np.sin(3*time*w) + 3*w*b3*np.cos(3*time*w)
#        alpha_2 = alpha_2/180*np.pi
#        Dalpha_2 = Dalpha_2/180*np.pi
#        
#        a0 =       13.46
#        a1 =       2.326
#        b1 =       8.468
#        a2 =       -50.6
#        b2 =      -27.37
#        a3 =       33.43
#        b3 =       23.4-adjust[ep-1]1
#        w =      0.7615
#        
#        alpha_1 = a0 + a1*np.cos(time*w) + b1*np.sin(time*w) + a2*np.cos(2*time*w) + b2*np.sin(2*time*w) + a3*np.cos(3*time*w) + b3*np.sin(3*time*w)
#        Dalpha_1 = -w*a1*np.sin(time*w) + w*b1*np.cos(time*w) - 2*w*a2*np.sin(2*time*w) + 2*w*b2*np.cos(2*time*w) - 3*w*a3*np.sin(3*time*w) + 3*w*b3*np.cos(3*time*w)
#        alpha_1 = alpha_1/180*np.pi
#        Dalpha_1 = Dalpha_1/180*np.pi
        
        """swimming"""
        alpha_2 =  np.cos(time+np.pi/4)*1;
        alpha_1 =  np.sin(time+np.pi/4)*1;
        Dalpha_2 = -np.sin(time+np.pi/4)*1;
        Dalpha_1 = np.cos(time+np.pi/4)*1;
        
        return alpha_1, alpha_2, Dalpha_1, Dalpha_2
    def __terminal(self):
        return 0
    def reset(self,beta0 = None, shape = None,straight = False):   # reset the environment setting
        self.pos, self.shape = self.__initialConfig(1,1)
        if beta0 != None:
            self.pos[-1] = beta0
        if shape is not None:
            self.shape = shape
        if straight:                 # used in policy tests
            self.shape = np.array([0,0])
        self.vel = np.zeros_like(self.pos)
        self.shapechange = np.zeros_like(self.shape)
        self.time = 0
        return self._get_obs()

    def _get_obs(self):                 # get the orientation (reltative to the targeted direction) and the shape
        return np.concatenate([np.array([self.pos[-1]]),self.shape])

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        from pyglet.gl import glRotatef, glPushMatrix


        def draw_lasting_circle(Viewer, radius=10, res=30, filled=True, **attrs):
            geom = rendering.make_circle(radius=radius, res=res, filled=filled)
            rendering._add_attrs(geom, attrs)
            Viewer.add_geom(geom)
            return geom
        
        def draw_lasting_line(Viewer, start, end, **attrs):
            geom = rendering.Line(start, end)
            rendering._add_attrs(geom, attrs)
            Viewer.add_geom(geom)
            return geom

        def make_ellipse(major=10, minor=5, res=30, filled=True):
            points = []
            for i in range(res):
                ang = 2*np.pi*i / res
                points.append((np.cos(ang)*major, np.sin(ang)*minor))
            if filled:
                return rendering.FilledPolygon(points)
            else:
                return rendering.PolyLine(points, True)


        def draw_ellipse(Viewer, major=10, minor=5, res=30, **attrs):
            geom = make_ellipse(major=major, minor=minor, res=res, filled=True)
            rendering._add_attrs(geom, attrs)
            Viewer.add_onetime(geom)
            return geom
        
        # ------------------------------------------------------------------------
        # size and position of the fish
        a,b,c = self.a,self.b,self.c
        x,y,theta = self.pos
        alpha_1, alpha_2 = self.shape
        theta = np.array([0, -alpha_1, alpha_2]) + theta
        x1 = x - np.cos(theta[0])*a[0] - np.cos(theta[1])*a[1]
        y1 = y - np.sin(theta[0])*a[0] - np.sin(theta[1])*a[1]
        x2 = x + np.cos(theta[0])*a[0] + np.cos(theta[2])*a[2]
        y2 = y + np.sin(theta[0])*a[0] + np.sin(theta[2])*a[2]
        x = np.array([x,x1,x2])
        y = np.array([y,y1,y2])
        # ------------------------------------------------------------------------
        from gym.envs.classic_control import rendering
        
        # create the image if it has not been done
        if self.viewer is None:
            self.viewer = rendering.Viewer(1000,200)
            background = draw_lasting_circle(self.viewer,radius=100, res=10)
            background.set_color(1.,1.,1.)
        
        # set viewer size
        bound = 4
        self.viewer.set_bounds(-bound+2,bound+4,-bound/4,bound/4)
        
        """draw two axes"""
        axisX = self.viewer.draw_line((-1000., 0), (1000., 0))
        axisY = self.viewer.draw_line((0,-1000.), (0,1000.))
        axisX.set_color(.5,.5,.5)
        axisY.set_color(.5,.5,.5)

        """draw a fish"""
        for i in range(3):
            link = draw_ellipse(self.viewer,major=self.a[i], minor=self.b[i], res=30, filled=True)
            lkTrans = rendering.Transform(rotation=theta[i],translation=(x[i],y[i]))
            link.add_attr(lkTrans)
            if i%3 == 0: link.set_color(.7, .1, .1)
            if i%3 == 1: link.set_color(.1, .7, .1)
            if i%3 == 2: link.set_color(.1, .1, .7)
        for i in range(2):
            eye = draw_ellipse(self.viewer,major=self.a[2]/8, minor=self.a[2]/8, res=30, filled=True)
            eyngle = theta[2]+np.pi/5.5*(i-.5)*2;
            eyeTrans = rendering.Transform(translation=(x[2]+np.cos(eyngle)*self.a[2]/2,y[2]+np.sin(eyngle)*self.a[2]/2))
            eye.add_attr(eyeTrans)
            eye.set_color(.6,.3,.4)
            
        # ------------------------------------------------------------------------
        # interpolate a smooth shape of the fish
        Npts = 7*2
        headl = 1.3
        facel = .6
        facew = 2.4
        headw = 2.6
        neckw = 2.3
        bodyw = 2.2
        waist = 2
        tailw = 1.7
        taill = 2.2
        
        referenceX = np.zeros((Npts,))
        referenceY = np.zeros((Npts,))
        
        referenceX[7], referenceY[7] = x2 + np.cos(theta[2])*a[2]*headl, y2 + np.sin(theta[2])*a[2]*headl
        
        referenceX[6], referenceY[6] = x2 + np.cos(theta[2])*a[2]*facel - np.sin(theta[2])*b[2]*facew, y2 + np.sin(theta[2])*a[2]*facel + np.cos(theta[2])*b[2]*facew
        referenceX[-6], referenceY[-6] = x2 + np.cos(theta[2])*a[2]*facel + np.sin(theta[2])*b[2]*facew, y2 + np.sin(theta[2])*a[2]*facel - np.cos(theta[2])*b[2]*facew
        
        referenceX[5], referenceY[5] = x2 - np.sin(theta[2])*b[2]*headw, y2 + np.cos(theta[2])*b[2]*headw
        referenceX[-5], referenceY[-5] = x2 + np.sin(theta[2])*b[2]*headw, y2 - np.cos(theta[2])*b[2]*headw
        
        referenceX[4], referenceY[4] = x[0] + np.cos(theta[0])*a[0] - np.sin((theta[0]+theta[2])/2)*(b[2]+b[0])*neckw/2, y[0] + np.sin(theta[0])*a[0] + np.cos((theta[0]+theta[2])/2)*(b[2]+b[0])*neckw/2
        referenceX[-4], referenceY[-4] = x[0] + np.cos(theta[0])*a[0] + np.sin((theta[0]+theta[2])/2)*(b[2]+b[0])*neckw/2, y[0] + np.sin(theta[0])*a[0] - np.cos((theta[0]+theta[2])/2)*(b[2]+b[0])*neckw/2
        
        referenceX[3], referenceY[3] = x[0] - np.sin(theta[0])*b[0]*bodyw, y[0] + np.cos(theta[0])*b[0]*bodyw
        referenceX[-3], referenceY[-3] = x[0] + np.sin(theta[0])*b[0]*bodyw, y[0] - np.cos(theta[0])*b[0]*bodyw
        
        referenceX[2], referenceY[2] = x[0] - np.cos(theta[0])*a[0] - np.sin((theta[0]+theta[1])/2)*(b[1]+b[0])*waist/2, y[0] - np.sin(theta[0])*a[0] + np.cos((theta[0]+theta[1])/2)*(b[1]+b[0])*waist/2
        referenceX[-2], referenceY[-2] = x[0] - np.cos(theta[0])*a[0] + np.sin((theta[0]+theta[1])/2)*(b[1]+b[0])*waist/2, y[0] - np.sin(theta[0])*a[0] - np.cos((theta[0]+theta[1])/2)*(b[1]+b[0])*waist/2
        
        referenceX[1], referenceY[1] = x1 - np.sin(theta[1])*b[1]*tailw, y1 + np.cos(theta[1])*b[1]*tailw
        referenceX[-1], referenceY[-1] = x1 + np.sin(theta[1])*b[1]*tailw, y1 - np.cos(theta[1])*b[1]*tailw
        
        referenceX[0], referenceY[0] = x1 - np.cos(theta[1])*a[1]*taill, y1 - np.sin(theta[1])*a[1]*taill
        
        referenceX = np.append(referenceX,referenceX[0])
        referenceY = np.append(referenceY,referenceY[0])
        
        from scipy.interpolate import CubicSpline
        p = np.linspace(0,1,num=Npts+1)
        cs = CubicSpline(p, np.stack([referenceX,referenceY]).T,bc_type='periodic')
        pnew = np.linspace(0,1,num=200)
        outout = cs(pnew)
        reference = []
        for i in range(np.size(outout,0)):
            reference.append((outout[i,0],outout[i,1]))
        # ------------------------------------------------------------------------
        
        fish = self.viewer.draw_polygon(reference, filled=False)
        fish.set_linewidth(2)
        fish.set_color(.5,.5,.5)
        
        """draw a trail behind the fish"""
        if self.oldpos is not None:
            trail = draw_lasting_circle(self.viewer, radius=0.015, res = 5)
            trTrans = rendering.Transform(translation=(np.sum(x)/3,np.sum(y)/3))
            trail.add_attr(trTrans)
            dx = self.pos[0]-self.oldpos[0]
            dy = self.pos[1]-self.oldpos[1]
            trail.set_color(0.3,0.3,0.65)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    def set_t_step(self, t_step):
        self.dt = t_step
# def angle_normalize(x):
#     return (((x+np.pi) % (2*np.pi)) - np.pi)
