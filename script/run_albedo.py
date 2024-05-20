
from tool import *
warnings.filterwarnings("ignore")


# #提取网格信息
# y=2001
# m=1
# time_x=time.time()
# print(y,m)
# checkpoint_lat=list(range(80,-90,-10))+[-89]
# result={}
# for lat in range(50,40,-1):
#     n=str(abs(lat))+'   '
#     print(n,end="",flush = True)
#     time1=time.time()
#     for lon in range(-180,180):
#         result[str(lat)+','+str(lon)]=get_grid_information(y,m,[lon,lat])
#         if lon%3==0:
#             if result[str(lat)+','+str(lon)][0]==1:
#                 print('*',end="",flush = True)
#             else:
#                 print('-',end="",flush = True)
#     time2=time.time()
#     print('    ',end="",flush = True)
#     print(int(time2-time1),end="",flush = True)
#     print('\n',end="",flush = True)
#     if lat in checkpoint_lat:
#         torch.save(result,f'/mnt/nvme1n1/hk/albedo/result_{y}_{m}_{lat}.pth')
#         result={}
# print(time.time()-time_x)
if __name__=='__main__':
    for y in [2001,2003,2005,2006]:
        
    # for y in [2007,2008,2009,2020]:    
        for m in range(1,13):
            print(f'------------------ {y}/{m} ------------------')
            for lat in tqdm(range(90,-90,-10)):
                cc=get_10lat_information(y,m,lat)
                cc.get_10lat_information()