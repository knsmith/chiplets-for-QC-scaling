OPENQASM 2.0;
include "qelib1.inc";
qreg q[430];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[0],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[13];
cx q[13],q[14];
cx q[14],q[15];
cx q[15],q[16];
cx q[16],q[17];
cx q[17],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[26];
cx q[26],q[27];
cx q[27],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[33];
cx q[33],q[34];
cx q[34],q[35];
cx q[35],q[36];
cx q[36],q[37];
cx q[37],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[43];
cx q[43],q[44];
cx q[44],q[45];
cx q[45],q[46];
cx q[46],q[47];
cx q[47],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[53];
cx q[53],q[54];
cx q[54],q[55];
cx q[55],q[56];
cx q[56],q[57];
cx q[57],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[63];
cx q[63],q[64];
cx q[64],q[65];
cx q[65],q[66];
cx q[66],q[67];
cx q[67],q[72];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[73];
cx q[73],q[74];
cx q[74],q[75];
cx q[75],q[76];
cx q[76],q[77];
cx q[77],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[83];
cx q[83],q[84];
cx q[84],q[85];
cx q[85],q[86];
cx q[86],q[87];
cx q[87],q[92];
cx q[92],q[91];
cx q[91],q[90];
cx q[90],q[93];
cx q[93],q[94];
cx q[94],q[95];
cx q[95],q[96];
cx q[96],q[97];
cx q[97],q[102];
cx q[102],q[101];
cx q[101],q[100];
cx q[100],q[103];
cx q[103],q[104];
cx q[104],q[105];
cx q[105],q[106];
cx q[106],q[107];
cx q[107],q[112];
cx q[112],q[111];
cx q[111],q[110];
cx q[110],q[113];
cx q[113],q[114];
cx q[114],q[115];
cx q[115],q[116];
cx q[116],q[117];
cx q[117],q[122];
cx q[122],q[121];
cx q[121],q[120];
cx q[120],q[123];
cx q[123],q[124];
cx q[124],q[125];
cx q[125],q[126];
cx q[126],q[127];
cx q[127],q[132];
cx q[132],q[131];
cx q[131],q[130];
cx q[130],q[133];
cx q[133],q[134];
cx q[134],q[135];
cx q[135],q[136];
cx q[136],q[137];
cx q[137],q[142];
cx q[142],q[141];
cx q[141],q[140];
cx q[140],q[143];
cx q[143],q[144];
cx q[144],q[145];
cx q[145],q[146];
cx q[146],q[147];
cx q[147],q[152];
cx q[152],q[151];
cx q[151],q[150];
cx q[150],q[153];
cx q[153],q[154];
cx q[154],q[155];
cx q[155],q[156];
cx q[156],q[157];
cx q[157],q[162];
cx q[162],q[161];
cx q[161],q[160];
cx q[160],q[163];
cx q[163],q[164];
cx q[164],q[165];
cx q[165],q[166];
cx q[166],q[167];
cx q[167],q[172];
cx q[172],q[171];
cx q[171],q[170];
cx q[170],q[173];
cx q[173],q[174];
cx q[174],q[175];
cx q[175],q[176];
cx q[176],q[177];
cx q[177],q[182];
cx q[182],q[181];
cx q[181],q[180];
cx q[180],q[183];
cx q[183],q[184];
cx q[184],q[185];
cx q[185],q[186];
cx q[186],q[187];
cx q[187],q[192];
cx q[192],q[191];
cx q[191],q[190];
cx q[190],q[193];
cx q[193],q[194];
cx q[194],q[195];
cx q[195],q[196];
cx q[196],q[197];
cx q[197],q[202];
cx q[202],q[201];
cx q[201],q[200];
cx q[200],q[203];
cx q[203],q[204];
cx q[204],q[205];
cx q[205],q[206];
cx q[206],q[207];
cx q[207],q[212];
cx q[212],q[211];
cx q[211],q[210];
cx q[210],q[213];
cx q[213],q[214];
cx q[214],q[215];
cx q[215],q[216];
cx q[216],q[217];
cx q[217],q[222];
cx q[222],q[221];
cx q[221],q[220];
cx q[220],q[223];
cx q[223],q[224];
cx q[224],q[225];
cx q[225],q[226];
cx q[226],q[227];
cx q[227],q[232];
cx q[232],q[231];
cx q[231],q[230];
cx q[230],q[233];
cx q[233],q[234];
cx q[234],q[235];
cx q[235],q[236];
cx q[236],q[237];
cx q[237],q[242];
cx q[242],q[241];
cx q[241],q[240];
cx q[240],q[243];
cx q[243],q[244];
cx q[244],q[245];
cx q[245],q[246];
cx q[246],q[247];
cx q[247],q[252];
cx q[252],q[251];
cx q[251],q[250];
cx q[250],q[253];
cx q[253],q[254];
cx q[254],q[255];
cx q[255],q[256];
cx q[256],q[257];
cx q[257],q[262];
cx q[262],q[261];
cx q[261],q[260];
cx q[260],q[263];
cx q[263],q[264];
cx q[264],q[265];
cx q[265],q[266];
cx q[266],q[267];
cx q[267],q[272];
cx q[272],q[271];
cx q[271],q[270];
cx q[270],q[273];
cx q[273],q[274];
cx q[274],q[275];
cx q[275],q[276];
cx q[276],q[277];
cx q[277],q[282];
cx q[282],q[281];
cx q[281],q[280];
cx q[280],q[283];
cx q[283],q[284];
cx q[284],q[285];
cx q[285],q[286];
cx q[286],q[287];
cx q[287],q[292];
cx q[292],q[291];
cx q[291],q[290];
cx q[290],q[293];
cx q[293],q[294];
cx q[294],q[295];
cx q[295],q[296];
cx q[296],q[297];
cx q[297],q[302];
cx q[302],q[301];
cx q[301],q[300];
cx q[300],q[303];
cx q[303],q[304];
cx q[304],q[305];
cx q[305],q[306];
cx q[306],q[307];
cx q[307],q[312];
cx q[312],q[311];
cx q[311],q[310];
cx q[310],q[313];
cx q[313],q[314];
cx q[314],q[315];
cx q[315],q[316];
cx q[316],q[317];
cx q[317],q[322];
cx q[322],q[321];
cx q[321],q[320];
cx q[320],q[323];
cx q[323],q[324];
cx q[324],q[325];
cx q[325],q[326];
cx q[326],q[327];
cx q[327],q[332];
cx q[332],q[331];
cx q[331],q[330];
cx q[330],q[333];
cx q[333],q[334];
cx q[334],q[335];
cx q[335],q[336];
cx q[336],q[337];
cx q[337],q[342];
cx q[342],q[341];
cx q[341],q[340];
cx q[340],q[343];
cx q[343],q[344];
cx q[344],q[345];
cx q[345],q[346];
cx q[346],q[347];
cx q[347],q[352];
cx q[352],q[351];
cx q[351],q[350];
cx q[350],q[353];
cx q[353],q[354];
cx q[354],q[355];
cx q[355],q[356];
cx q[356],q[357];
cx q[357],q[362];
cx q[362],q[361];
cx q[361],q[360];
cx q[360],q[363];
cx q[363],q[364];
cx q[364],q[365];
cx q[365],q[366];
cx q[366],q[367];
cx q[367],q[372];
cx q[372],q[371];
cx q[371],q[370];
cx q[370],q[373];
cx q[373],q[374];
cx q[374],q[375];
cx q[375],q[376];
cx q[376],q[377];
cx q[377],q[382];
cx q[382],q[381];
cx q[381],q[380];
cx q[380],q[383];
cx q[383],q[384];
cx q[384],q[385];
cx q[385],q[386];
cx q[386],q[387];
cx q[387],q[392];
cx q[392],q[391];
cx q[391],q[390];
cx q[390],q[393];
cx q[393],q[394];
cx q[394],q[395];
cx q[395],q[396];
cx q[396],q[397];
cx q[397],q[402];
cx q[402],q[401];
cx q[401],q[400];
cx q[400],q[403];
cx q[403],q[404];
cx q[404],q[405];
cx q[405],q[406];
cx q[406],q[407];
cx q[407],q[412];
cx q[412],q[411];
cx q[411],q[410];
cx q[410],q[413];
cx q[413],q[414];
cx q[414],q[415];
cx q[415],q[416];
cx q[416],q[417];
cx q[417],q[422];
cx q[422],q[421];
cx q[421],q[420];
cx q[420],q[423];
cx q[423],q[424];
cx q[424],q[425];
cx q[425],q[426];
cx q[426],q[427];
