#from .MapGenarator import RougeMapGenerator
import MapGenarator
import numpy as np
import random

### parameters
N = 50
mapH = 20
mapW = 40
rN = 4
roomH = 3
roomW = 3


print('Start')
print('Map Generating...')

MG = MapGenarator.RougeMapGenerator(
    N,
    mapH,
    mapW,
    rN,
    False,
    roomH,
    roomW,
    False
)
MG.Generate()

print('Generate Done!')
while True:
    key = input()
    if key == 'a':
        mp, si, sj, pi, pj = MG.GetMap(
            random.randrange(N)
        )

        for i in range(mapH):
            s = ''
            for j in range(mapW):
                if (i, j) == (si, sj):
                    s+='G'
                elif (i, j) == (pi, pj):
                    s+='S'
                else:
                    #s+=str(mp[i][j])
                    if mp[i][j] > 0:
                        s+='.'
                    else:
                        s+='#'
            print(s)
    else:
        print('End')
        break


