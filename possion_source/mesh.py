import numpy as np
import torch

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def ellipticMap(x,y,h,tol):
	eps=2.2e-16
	[ny,nx]=x.shape
	ite=1
	A=np.ones([ny-2,nx-2]); B=A; C=A;
	Err=[]
	while True:
		X=(A*(x[2:,1:-1]+x[0:-2,1:-1])+C*(x[1:-1,2:]+x[1:-1,0:-2])-\
		  B/2*(x[2:,2:]+x[0:-2,0:-2]-x[2:,0:-2]-x[0:-2,2:]))/2/(A+C)
		Y=(A*(y[2:,1:-1]+y[0:-2,1:-1])+C*(y[1:-1,2:]+y[1:-1,0:-2])-\
		  B/2*(y[2:,2:]+y[0:-2,0:-2]-y[2:,0:-2]-y[0:-2,2:]))/2/(A+C)
		err=np.max(np.max(np.abs(x[1:-1,1:-1]-X)))+\
			np.max(np.max(np.abs(y[1:-1,1:-1]-Y)))
		#print('error at this iteration'+'===>'+str(err))
		Err.append(err)
		x[1:-1,1:-1]=X; y[1:-1,1:-1]=Y
		A=((x[1:-1,2:]-x[1:-1,0:-2])/2/h)**2+\
		  ((y[1:-1,2:]-y[1:-1,0:-2])/2/h)**2+eps
		B=(x[2:,1:-1]-x[0:-2,1:-1])/2/h*\
		  (x[1:-1,2:]-x[1:-1,0:-2])/2/h+\
		  (y[2:,1:-1]-y[0:-2,1:-1])/2/h*\
		  (y[1:-1,2:]-y[1:-1,0:-2])/2/h+eps
		C=((x[2:,1:-1]-x[0:-2,1:-1])/2/h)**2+\
		  ((y[2:,1:-1]-y[0:-2,1:-1])/2/h)**2+eps
		if err<tol:
			print('The mesh generation reaches covergence!')
			break; pass
		if ite>50000:
			print('The mesh generation not reaches covergence '+\
				  'within 50000 iterations! The current resdiual is ')
			print(err)
			break; pass
		ite=ite+1
	return x, y

class Mesh():
	"""docstring for Mesh"""
	def __init__(self,leftX,leftY,rightX,rightY,lowX,lowY,upX,upY,
		         h,tolMesh=1e-8):
		self.h=h
		self.tolMesh=tolMesh
		self.ny=leftX.shape[0]; self.nx=upX.shape[0]
		# Prellocate the physical domain
		#Left->Right->Low->Up
		self.x=np.zeros((self.ny,self.nx))
		self.y=np.zeros((self.ny,self.nx))
		self.x[:,0]=leftX; self.y[:,0]=leftY
		self.x[:,-1]=rightX; self.y[:,-1]=rightY
		self.x[0,:]=lowX; self.y[0,:]=lowY
		self.x[-1,:]=upX; self.y[-1,:]=upY
		self.x,self.y=ellipticMap(self.x,self.y,self.h,self.tolMesh)
		# Define the ref domain
		eta,xi=np.meshgrid(np.linspace(0,self.ny-1,self.ny),
	               np.linspace(0,self.nx-1,self.nx),
	               sparse=False,indexing='ij')
		self.xi=xi*h; self.eta=eta*h;
		dxdxi_ho_internal=(-self.x[:,4:]+8*self.x[:,3:-1]-\
			           8*self.x[:,1:-3]+self.x[:,0:-4])/12/self.h
		dydxi_ho_internal=(-self.y[:,4:]+8*self.y[:,3:-1]-\
			           8*self.y[:,1:-3]+self.y[:,0:-4])/12/self.h
		dxdeta_ho_internal=(-self.x[4:,:]+8*self.x[3:-1,:]-\
			            8*self.x[1:-3,:]+self.x[0:-4,:])/12/self.h
		dydeta_ho_internal=(-self.y[4:,:]+8*self.y[3:-1,:]-\
			            8*self.y[1:-3,:]+self.y[0:-4,:])/12/self.h

		dxdxi_ho_left=(-11*self.x[:,0:-3]+18*self.x[:,1:-2]-\
			           9*self.x[:,2:-1]+2*self.x[:,3:])/6/self.h
		dxdxi_ho_right=(11*self.x[:,3:]-18*self.x[:,2:-1]+\
			           9*self.x[:,1:-2]-2*self.x[:,0:-3])/6/self.h
		dydxi_ho_left=(-11*self.y[:,0:-3]+18*self.y[:,1:-2]-\
			           9*self.y[:,2:-1]+2*self.y[:,3:])/6/self.h
		dydxi_ho_right=(11*self.y[:,3:]-18*self.y[:,2:-1]+\
			           9*self.y[:,1:-2]-2*self.y[:,0:-3])/6/self.h
		dxdeta_ho_low=(-11*self.x[0:-3,:]+18*self.x[1:-2,:]-\
			           9*self.x[2:-1,:]+2*self.x[3:,:])/6/self.h
		dxdeta_ho_up=(11*self.x[3:,:]-18*self.x[2:-1,:]+\
			           9*self.x[1:-2,:]-2*self.x[0:-3,:])/6/self.h
		dydeta_ho_low=(-11*self.y[0:-3,:]+18*self.y[1:-2,:]-\
			           9*self.y[2:-1,:]+2*self.y[3:,:])/6/self.h
		dydeta_ho_up=(11*self.y[3:,:]-18*self.y[2:-1,:]+\
			           9*self.y[1:-2,:]-2*self.y[0:-3,:])/6/self.h
		self.dxdxi_ho=np.zeros(self.x.shape)
		self.dxdxi_ho[:,2:-2]=dxdxi_ho_internal
		self.dxdxi_ho[:,0:2]=dxdxi_ho_left[:,0:2]
		self.dxdxi_ho[:,-2:]=dxdxi_ho_right[:,-2:]

		self.dydxi_ho=np.zeros(self.y.shape)
		self.dydxi_ho[:,2:-2]=dydxi_ho_internal
		self.dydxi_ho[:,0:2]=dydxi_ho_left[:,0:2]
		self.dydxi_ho[:,-2:]=dydxi_ho_right[:,-2:]

		self.dxdeta_ho=np.zeros(self.x.shape)
		self.dxdeta_ho[2:-2,:]=dxdeta_ho_internal
		self.dxdeta_ho[0:2,:]=dxdeta_ho_low[0:2,:]
		self.dxdeta_ho[-2:,:]=dxdeta_ho_up[-2:,:]

		self.dydeta_ho=np.zeros(self.y.shape)
		self.dydeta_ho[2:-2,:]=dydeta_ho_internal
		self.dydeta_ho[0:2,:]=dydeta_ho_low[0:2,:]
		self.dydeta_ho[-2:,:]=dydeta_ho_up[-2:,:]

		self.J_ho=self.dxdxi_ho*self.dydeta_ho-\
		          self.dxdeta_ho*self.dydxi_ho
		self.Jinv_ho=1/self.J_ho
			






