from tool import *
warnings.filterwarnings("ignore")
if __name__=='__main__':
    for y in range(2011,2021):
        for m in range(1,13):
            _=monthly_snow_cover(y,m,True)