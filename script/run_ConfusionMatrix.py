from tool import *
if __name__=='__main__':

    for y in [2018]:    
        for m in range(6,13):
            print(f'------------------ {y}/{m} ------------------')
            for lat_10 in tqdm(range(90,-90,-10)):
                get_10lat_confusionmatrix(y,m,lat_10)