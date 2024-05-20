from tool import *
warnings.filterwarnings("ignore")
if __name__=='__main__':
    # for y in [2001,2003,2004,2005]:
    for y in [2006,2007,2008,2009,2020]:
        print(y)
        time.sleep(1)
        result={}
        for lat in tqdm(range(90,-90,-1)):
            for lon in range(-180,180):
                result[str(lat)+','+str(lon)]=get_land_area(y,[lon,lat])
        torch.save(result,f"/data/hk/albedo/landcover/{y}_landcover/{y}_landarea.pth")