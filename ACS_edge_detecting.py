import numpy
import random
import math
import matplotlib.image as img
from matplotlib import pyplot as plt
import matplotlib

img=img.imread("cb.jpg")
img_out = "cb_after.jpg"
_reshold = 0.135

def absolute(a):
    return a
    res = 0
    for i in a:
        res += i*i
    return numpy.sqrt(res)
    

def normalization(img):
    A_return = []
    Vmax = 0.0
    for i in range(len(img[1,:])):
        lista = []
        for j in range(0, len(img[:,1])):
            val = 0
            if i > 0 and j > 0 and i < len(img[1,:]) - 1 and j < len(img[:,1]) - 1:
                val = absolute(img[i-1, j-1] - img[i+1,j+1]) + absolute(img[i-1,j] - img[i+1,j]) + absolute(img[i-1, j+1] - img[i+1,j-1]) + absolute(img[i, j-1] - img[i,j+1])
                #val = abs(img[i-1, j-1] - img[i+1,j+1]) + abs(img[i-1,j] - img[i+1,j]) + abs(img[i-1, j+1] - img[i+1,j-1]) + abs(img[i, j-1] - img[i,j+1])
            lista.append(val)
            Vmax = max(Vmax, val)
        A_return.append(lista)


    for i in range(1,len(img[1,:])-1):
        for j in range(1,len(img[:,1])-1):
            A_return[i][j] /= Vmax
    return A_return



class ACO:
    def __init__(self,img, br_iter,br_ant,br_step,alpha,beta,phi,p,q0,tau0,threshold):
        
        
        self.img = img
        self.br_iter = br_iter
        self.br_ant = br_ant
        self.br_step = br_step
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.tau0 = tau0
        self.q0 = q0
        self.p = p
        self.threshHold = threshold

        self.Pheromones = []
        for i in range(len(img[:,1])):
            listA = []
            for j in range(len(img[1,:])):
                listA.append(self.tau0)
            self.Pheromones.append(listA)
    
            
        self.delta_tau = []

        self.iter = []
        for i in range(len(img[:,1])):
            lista = []
            for j in range(len(img[1,:])):
                lista.append(0)
            self.delta_tau.append(lista)
            self.iter.append(lista)

        self.Informations = normalization(img)

        self.Routes = []
        self.totalHeuristic = []
        self.depth = []
        self.check = []
        for i in range(br_ant):
            self.Routes.append([[random.randint(1, len(img[1,:])-1), random.randint(1, len(img[:,1])-1)]])
            self.totalHeuristic.append(0.0)
            self.depth.append(0)
            self.check.append(True)


    def run(self):
        for iter in range(self.br_iter):
            for step in range(self.br_step):
                for antNum in range(self.br_ant):
                    self.local_update(antNum,step)
            self.global_update(iter)


    def local_update(self,k,l):
        if not self.check[k]:
            return
        i0=self.Routes[k][l][0]
        j0=self.Routes[k][l][1]
        xMinLim=i0-1
        xMaxLim=i0+1
        yMinLim=j0-1
        yMaxLim=j0+1
        
        if i0==0: 
            xMinLim==0
        if j0==0:
            yMinLim==0
        if i0>=len(self.img[1,:])-1:
            xMaxLim=len(self.img[1,:])-1
        if j0>=len(self.img[:,1])-1:
            yMaxLim=len(self.img[:,1])-1
        
        neighbourhood=[]
        
        for i in range(xMinLim, xMaxLim+1): 
            for j in range(yMinLim, yMaxLim+1):
                if (i!=i0 or j!=j0):
                    false = 0
                    for positions in self.Routes[k]:
                        if positions[0]==i and positions[1]==j:
                            false = 1
                            break
                    if false == 0:
                        neighbourhood.append([i,j])
        random.shuffle(neighbourhood)
        m = i0
        n = j0
        val = []
        if len(neighbourhood) == 0:
            m=i0
            n=j0
            
        else:

            id = -1
            maxNeighbor = 0
            sumVal = 0.0
            for i in range(len(neighbourhood)):
                p = float(pow(self.Pheromones[neighbourhood[i][0]][neighbourhood[i][1]], self.alpha)) * \
                    float(pow(self.Informations[neighbourhood[i][0]][neighbourhood[i][1]], self.beta))
                val.append(p)
                sumVal += p
                id += 1
                if maxNeighbor == -1 or val[maxNeighbor] < p:
                    maxNeighbor = id
            if sumVal == 0:
                m = i0
                n = j0
            else:
                u = random.uniform(0.0,1.0)
                if u <= self.q0:
                    m = neighbourhood[maxNeighbor][0]
                    n = neighbourhood[maxNeighbor][1]
                else:
                    u0 = random.uniform(0.0,1.0)
                    for i in range(len(neighbourhood)):
                        u0 -= val[i]/sumVal
                        if i == len(neighbourhood) - 1:
                            u0 = 0
                        if u0 <= 0:
                            m = neighbourhood[i][0]
                            n = neighbourhood[i][1]
                            break

        if i0 != m or j0 != n:
            self.Pheromones[m][n]=(1-self.phi)*self.Pheromones[m][n]+self.phi*self.tau0
            self.depth[k] += 1
            
            
            self.totalHeuristic[k] += self.Informations[m][n]
            
            
        else:
            self.check[k] = False
        self.Routes[k].append([m,n])

        
    def global_update(self, iter):
        #decay
        for i in range(self.br_ant):
            if self.depth[i] == 0:
                self.check[i] = True
                continue
            
                
            avgHeuristic = self.totalHeuristic[i] / self.depth[i]
            
            
            for j in range(1, self.depth[i]+1):
                route = self.Routes[i][j]
                if self.iter[route[0]][route[1]] != iter:
                    self.iter[route[0]][route[1]] = iter
                    self.Pheromones[route[0]][route[1]] *= 1 - self.p
                self.Pheromones[route[0]][route[1]] += self.p * avgHeuristic
                self.totalHeuristic[i] = 0.0
            curX = self.Routes[i][len(self.Routes[i]) - 1][0]
            curY = self.Routes[i][len(self.Routes[i]) - 1][1]
            self.Routes[i] = [[random.randint(1, len(img[1,:])-1),random.randint(1, len(img[:,1])-1)]]
            self.depth[i] = 0
            self.check[i] = True


                
    def showImage(self):
        FinalImage=[]
        
        for i in range(0, len(self.img[1,:])):
            lista=[]
            for j in range(0, len(self.img[:,1])):
                if self.Pheromones[i][j] < _reshold:
                    lista.append([1.0,1.0,1.0])
                    
                else:
                    lista.append([0.0,0.0,0.0])
                    
            FinalImage.append(lista)
                        
        plt.imshow(FinalImage, interpolation='nearest')
        plt.show()
        matplotlib.image.imsave(img_out, FinalImage)
       
        

# N=10, L=50, K=5000, alpha=1.0, beta=2.0, phi=0.05, tau0=0.1, q0=0.2 p=0.1
a = ACO(img,20,1024,100,1.0,1.0,0.05,0.1,0.4,0.1,0.1)
a.run()
a.showImage()
