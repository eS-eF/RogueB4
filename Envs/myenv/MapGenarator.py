'''

map 生成システム

仕様 :

・openAI のインターフェースを使う（openAI の env として使えるように）

・学習開始時、マップを num_map 個自動生成し、エピソード毎に生成したマップのうち１つを渡す

・生成するマップについて ------

・サイズは小さく、可能なら決定できるように
・部屋数は ~5 くらいで（最低でも 3）
・形状を指定できるようにしたい（後戻りが必要になる形を指定できるように）
・通路、部屋、階段のみ
・01 多次元vector で渡す、H*W*3

---------------------------

・action_space は 8? 4でも良さそうではある
・reward は外部から設定できて欲しい ゴール到達時は 1.0 で統一したい
・HP など余計なものは廃止
・マップは行動に際して差分のみ更新、ゴールは座標で覚えておく
・壁と未知の領域で数値を分けてみる？


（同時進行）
コードのリファクタリング
実験コードが汚すぎるので本体、保存、データなど分けるように


'''
import numpy as np
import random
import copy

class RougeMapGenerator:
    def __init__(self, Num_map, _H, _W, Num_room, Type_fix, Room_min_h, Room_min_w, map_type_flag):
        self.num_map = Num_map
        self.H = _H
        self.W = _W
        self.num_room = Num_room
        self.type_fix = Type_fix

        self.room_min_h = Room_min_h
        self.room_min_w = Room_min_w

        self.maps = np.zeros((self.num_map, self.H, self.W), dtype=np.uint8) #マップを保存、ここではone-hotではない
        # 0:壁、1:床
        self.rooms : list[list[RoomRect]] = [
            [] for _ in range(self.num_map)
        ]

        self.rects : list[RoomRect] = []


    '''
    配列をジェネレータの内部に作る。外部からはインデックスで i 番目のマップにアクセスする
    '''
    def Generate(self):
        # まず、マップをいくつかの長方形に分割する
        # 既存の長方形について、rect[i] には左上の座標、大きさ h・w を記録しておく
        # 分割に伴って更新すれば良い
        # rect は部屋数オーダーにしかしない

        for Map_id in range(self.num_map):
            self.rects.clear()

            self.rects.append(RoomRect((0,0), self.H, self.W)) #初期値

            for i in range(self.num_room - 1):
                #分割する
                #j = random.randrange(len(self.rects))
                seps = []
                for j in range(i+1):
                    #存在する部屋について、分割可能か判定
                    #縦方向
                    if self.rects[j].H >= 4 + 2 * self.room_min_h:
                        seps.append((j, 'H'))
                    #横方向
                    if self.rects[j].W >= 4 + 2 * self.room_min_w:
                        seps.append((j, 'W'))
                
                #分割をランダムに選択する
                if len(seps) == 0:
                    print('No separation : Room Num = ', i + 1)
                    break

                sep_id = random.randrange(len(seps))
                r_sep = seps[sep_id]
                rect_id = r_sep[0]

                rect_sep = self.rects[rect_id]

                #分割を行う
                #分割可能な位置からランダムに選択
                #部屋の最小サイズが残るように
                pos_separate = -1
                newrect : RoomRect

                if r_sep[1] == 'H':
                    
                    pos_separate = random.randrange(
                        2 + self.room_min_h,
                        rect_sep.H - 1 - self.room_min_h
                    ) + rect_sep.si
                    newrect = RoomRect(
                        (pos_separate, rect_sep.sj),
                        rect_sep.H - pos_separate + rect_sep.si,
                        rect_sep.W
                    )
                    ### 2+room_min_h <= rect_sep.H - pos_separate
                    self.rects[rect_id].H = pos_separate - rect_sep.si
                else:
                    pos_separate = random.randrange(
                        2 + self.room_min_w,
                        rect_sep.W - 1 - self.room_min_w
                    ) + rect_sep.sj
                    newrect = RoomRect(
                        (rect_sep.si, pos_separate),
                        rect_sep.H,
                        rect_sep.W - pos_separate + rect_sep.sj
                    )
                    self.rects[rect_id].W = pos_separate - rect_sep.sj
                
                ### 新しい長方形の追加
                self.rects.append(newrect)

                '''
                print('-----Separate ', i+1, ' -------')
                for rect in self.rects:
                    rect.debug_display()
                '''

            ### 長方形の準備完了
            for i in range(self.num_room):
                ### 部屋をランダム位置に生成
                cur_rect = self.rects[i]
                ### 部屋サイズをランダムに決定
                room_size_h = (random.randrange(
                    self.room_min_h,
                    (cur_rect.H - 2) + 1
                ) + random.randrange(
                    self.room_min_h,
                    (cur_rect.H - 2) + 1
                )) // 2

                room_size_w = (random.randrange(
                    self.room_min_w,
                    (cur_rect.W - 2) + 1
                ) + random.randrange(
                    self.room_min_w,
                    (cur_rect.W - 2) + 1
                )) // 2

                ### 部屋の位置をランダムに決定

                pos_i = random.randrange(
                    1,
                    (cur_rect.H - 1) - (room_size_h - 1)
                ) + cur_rect.si

                pos_j = random.randrange(
                    1,
                    (cur_rect.W - 1) - (room_size_w - 1)
                ) + cur_rect.sj

                ### パラメータの設定
                self.rects[i].room_si = pos_i
                self.rects[i].room_sj = pos_j
                self.rects[i].room_H = room_size_h
                self.rects[i].room_W = room_size_w
            
            '''
            print('-----Debug roomrects-----')
            for rect in self.rects:
                rect.debug_display()
            '''
            
            
            ### グラフを作成（通路生成の準備）

            Edges = []

            for i in range(self.num_room):
                for j in range(self.num_room):
                    if i >= j:
                        continue
                    recti = self.rects[i]
                    rectj = self.rects[j]
                    ### 領域が接しているかの判定
                    ### 縦, i が上
                    if recti.si + recti.H == rectj.si:
                        Edges.append((i, j, 0, rectj.si))
                        continue

                    ### 縦、 j が上
                    if rectj.si + rectj.H == recti.si:
                        Edges.append((i, j, 1, recti.si))
                        continue

                    ### 横、i が左
                    if recti.sj + recti.W == rectj.sj:
                        Edges.append((i, j, 2, rectj.sj))
                        continue

                    ### 横、j が左
                    if rectj.sj + rectj.W == recti.sj:
                        Edges.append((i, j, 3, recti.sj))
            ### 辺をシャッフルし、全域木を作る

            random.shuffle(Edges)
            RestEdge = []
            UseEdge = []
            uf = UnionFind(self.num_room)
            for i in range(len(Edges)):
                x, y, _, _ = Edges[i]
                if uf.same(x, y):
                    RestEdge.append(Edges[i])
                else:
                    UseEdge.append(Edges[i])
                    uf.unite(x, y)
            
            '''
            print('-----Debug 1------')
            print('Edges = ', Edges)
            print(uf.size(0), self.num_room, UseEdge)
            '''
            assert(uf.size(0) == self.num_room)

            ### 残った辺をもう一度シャッフルし、ランダムな本数採用する
            rest_num = random.randrange(0, len(RestEdge)) if len(RestEdge) > 0 else 0
            random.shuffle(RestEdge)
            for i in range(rest_num):
                UseEdge.append(RestEdge[i])

            ### useedgeにある辺を通路として接続する

            ### 部屋の１辺から通路を伸ばし、境界で横に接続する
            ### と、その前にマップへの書き込みを開始する
            ### まず、部屋を描画する

            for k in range(self.num_room):
                for i in range(self.rects[k].room_H):
                    for j in range(self.rects[k].room_W):
                        self.maps[Map_id][self.rects[k].room_si + i][self.rects[k].room_sj + j] = 1

            for x, y, f, bord in UseEdge:
                rectx = self.rects[x]
                recty = self.rects[y]

                if f == 0:
                    ### x が上
                    x_passage = random.randrange(
                        rectx.room_sj,
                        rectx.room_sj + rectx.room_W
                    )
                    
                    y_passage = random.randrange(
                        recty.room_sj,
                        recty.room_sj + recty.room_W
                    )

                    ### passage を境界で接続する
                    for i in range(
                        rectx.room_si + rectx.room_H,
                        bord):
                        self.maps[Map_id][i][x_passage] = 2
                    for i in range(
                        bord,
                        recty.room_si):
                        self.maps[Map_id][i][y_passage] = 2
                    
                    for j in range(
                        min(x_passage, y_passage),
                        max(x_passage, y_passage) + 1):
                        self.maps[Map_id][bord][j] = 2
                elif f == 1:
                    ### y が上
                    x_passage = random.randrange(
                        rectx.room_sj,
                        rectx.room_sj + rectx.room_W
                    )
                    
                    y_passage = random.randrange(
                        recty.room_sj,
                        recty.room_sj + recty.room_W
                    )

                    ### passage を境界で接続する
                    for i in range(
                        bord,
                        rectx.room_si):
                        self.maps[Map_id][i][x_passage] = 2
                    for i in range(
                        recty.room_si + recty.room_H,
                        bord):
                        self.maps[Map_id][i][y_passage] = 2
                    
                    for j in range(
                        min(x_passage, y_passage),
                        max(x_passage, y_passage) + 1):
                        self.maps[Map_id][bord][j] = 2

                elif f == 2:
                    ### x が左
                    x_passage = random.randrange(
                        rectx.room_si,
                        rectx.room_si + rectx.room_H
                    )
                    y_passage = random.randrange(
                        recty.room_si,
                        recty.room_si + recty.room_H
                    )

                    ### passageを境界で接続する
                    for j in range(
                        rectx.room_sj + rectx.room_W,
                        bord):
                        self.maps[Map_id][x_passage][j] = 2
                    
                    for j in range(
                        bord,
                        recty.room_sj):
                        self.maps[Map_id][y_passage][j] = 2

                    for i in range(
                        min(x_passage, y_passage),
                        max(x_passage, y_passage) + 1):
                        self.maps[Map_id][i][bord] = 2

                else:
                    ### y が左
                    x_passage = random.randrange(
                        rectx.room_si,
                        rectx.room_si + rectx.room_H
                    )
                    y_passage = random.randrange(
                        recty.room_si,
                        recty.room_si + recty.room_H
                    )

                    ### passageを境界で接続する
                    for j in range(
                        bord,
                        rectx.room_sj):
                        self.maps[Map_id][x_passage][j] = 2
                    
                    for j in range(
                        recty.room_sj + recty.room_W,
                        bord):
                        self.maps[Map_id][y_passage][j] = 2

                    for i in range(
                        min(x_passage, y_passage),
                        max(x_passage, y_passage) + 1):
                        self.maps[Map_id][i][bord] = 2


            ### 最後に、階段位置の生成候補を記録しておく
            ### 部屋の情報を残しておく（改善の余地あり？）
            self.rooms.append([])
            for rect in self.rects:
                self.rooms[Map_id].append(rect)








    def GetMap(self, idx):
        ### idx 番目のマップを渡す
        ### 階段の位置、初期位置を自動生成する

        ret_map = copy.deepcopy(self.maps[idx])
        ret_rooms = copy.deepcopy(self.rooms[idx])

        ### 階段
        stair_i = -1
        stair_j = -1
        
        room_id = random.randrange(
            0,
            len(ret_rooms)
        )
        stair_i = random.randrange(
            ret_rooms[room_id].room_si,
            ret_rooms[room_id].room_si + ret_rooms[room_id].room_H
        )
        stair_j = random.randrange(
            ret_rooms[room_id].room_sj,
            ret_rooms[room_id].room_sj + ret_rooms[room_id].room_W
        )



        player_i = -1
        player_j = -1
        room_id = random.randrange(
            0,
            len(ret_rooms)
        )

        player_i = random.randrange(
            ret_rooms[room_id].room_si,
            ret_rooms[room_id].room_si + ret_rooms[room_id].room_H
        )
        player_j = random.randrange(
            ret_rooms[room_id].room_sj,
            ret_rooms[room_id].room_sj + ret_rooms[room_id].room_W
        )

        ### 座標が被った場合、再抽選する
        while (player_i == stair_i and player_j == stair_j):
            room_id = random.randrange(
            0,
            len(ret_rooms)
            )

            player_i = random.randrange(
                ret_rooms[room_id].room_si,
                ret_rooms[room_id].room_si + ret_rooms[room_id].room_H
            )
            player_j = random.randrange(
                ret_rooms[room_id].room_sj,
                ret_rooms[room_id].room_sj + ret_rooms[room_id].room_W
            )
        
        return (ret_map, stair_i, stair_j, player_i, player_j)






class RoomRect:
    def __init__(self, Start_point, _H, _W):
        self.H = _H
        self.W = _W
        self.si = Start_point[0]
        self.sj = Start_point[1]

        self.room_H = -1
        self.room_W = -1
        self.room_si = -1
        self.room_sj = -1
    
    def area(self):
        return self.H * self.W
    
    def debug_display(self):
        print('room info : ')
        print('(si, sj) = ', self.si, self.sj, 'H = ', self.H, 'W = ', self.W)
        print('room : (rsi, rsj) = ', self.room_si, self.room_sj, 'rH = ', self.room_H, 'rW = ', self.room_W)
        

class UnionFind:
    def __init__(self, N):
        self.p = [-1] * N
 
    def root(self, x):
        while self.p[x] >= 0:
            x = self.p[x]
        return x
 
    def same(self, x, y):
        return self.root(x) == self.root(y)
 
    def unite(self, x, y):
        x = self.root(x)
        y = self.root(y)
        if x == y:
            return
        p = self.p
        if p[x] > p[y]:
            p[y] += p[x]
            p[x] = y
        else:
            p[x] += p[y]
            p[y] = x
 
    def size(self, x):
        return -self.p[self.root(x)]
