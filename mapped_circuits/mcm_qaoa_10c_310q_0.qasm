OPENQASM 2.0;
include "qelib1.inc";
qreg q[310];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(-pi) q[1];
x q[1];
x q[2];
cx q[2],q[1];
cx q[0],q[3];
rz(-pi) q[0];
x q[0];
cx q[1],q[0];
cx q[3],q[4];
rz(-pi) q[3];
x q[3];
cx q[0],q[3];
cx q[4],q[5];
rz(-pi) q[4];
x q[4];
cx q[3],q[4];
cx q[5],q[6];
rz(-pi) q[5];
x q[5];
cx q[4],q[5];
cx q[6],q[7];
rz(-pi) q[6];
x q[6];
cx q[5],q[6];
cx q[7],q[12];
cx q[12],q[11];
cx q[11],q[10];
rz(-pi) q[11];
x q[11];
rz(-pi) q[12];
x q[12];
rz(-pi) q[7];
x q[7];
cx q[6],q[7];
cx q[7],q[12];
cx q[12],q[11];
cx q[10],q[13];
rz(-pi) q[10];
x q[10];
cx q[11],q[10];
cx q[13],q[14];
rz(-pi) q[13];
x q[13];
cx q[10],q[13];
cx q[14],q[15];
rz(-pi) q[14];
x q[14];
cx q[13],q[14];
cx q[15],q[16];
rz(-pi) q[15];
x q[15];
cx q[14],q[15];
cx q[16],q[17];
rz(-pi) q[16];
x q[16];
cx q[15],q[16];
cx q[17],q[22];
rz(-pi) q[17];
x q[17];
cx q[16],q[17];
cx q[22],q[21];
cx q[21],q[20];
rz(-pi) q[21];
x q[21];
rz(-pi) q[22];
x q[22];
cx q[17],q[22];
cx q[22],q[21];
cx q[20],q[23];
rz(-pi) q[20];
x q[20];
cx q[21],q[20];
cx q[23],q[24];
rz(-pi) q[23];
x q[23];
cx q[20],q[23];
cx q[24],q[25];
rz(-pi) q[24];
x q[24];
cx q[23],q[24];
cx q[25],q[26];
rz(-pi) q[25];
x q[25];
cx q[24],q[25];
cx q[26],q[27];
rz(-pi) q[26];
x q[26];
cx q[25],q[26];
cx q[27],q[32];
rz(-pi) q[27];
x q[27];
cx q[26],q[27];
cx q[32],q[31];
cx q[31],q[30];
rz(-pi) q[31];
x q[31];
rz(-pi) q[32];
x q[32];
cx q[27],q[32];
cx q[32],q[31];
cx q[30],q[33];
rz(-pi) q[30];
x q[30];
cx q[31],q[30];
cx q[33],q[34];
rz(-pi) q[33];
x q[33];
cx q[30],q[33];
cx q[34],q[35];
rz(-pi) q[34];
x q[34];
cx q[33],q[34];
cx q[35],q[36];
rz(-pi) q[35];
x q[35];
cx q[34],q[35];
cx q[36],q[37];
rz(-pi) q[36];
x q[36];
cx q[35],q[36];
cx q[37],q[42];
rz(-pi) q[37];
x q[37];
cx q[36],q[37];
cx q[42],q[41];
cx q[41],q[40];
rz(-pi) q[41];
x q[41];
rz(-pi) q[42];
x q[42];
cx q[37],q[42];
cx q[42],q[41];
cx q[40],q[43];
rz(-pi) q[40];
x q[40];
cx q[41],q[40];
cx q[43],q[44];
rz(-pi) q[43];
x q[43];
cx q[40],q[43];
cx q[44],q[45];
rz(-pi) q[44];
x q[44];
cx q[43],q[44];
cx q[45],q[46];
rz(-pi) q[45];
x q[45];
cx q[44],q[45];
cx q[46],q[47];
rz(-pi) q[46];
x q[46];
cx q[45],q[46];
cx q[47],q[52];
rz(-pi) q[47];
x q[47];
cx q[46],q[47];
cx q[52],q[51];
cx q[51],q[50];
rz(-pi) q[51];
x q[51];
rz(-pi) q[52];
x q[52];
cx q[47],q[52];
cx q[52],q[51];
cx q[50],q[53];
rz(-pi) q[50];
x q[50];
cx q[51],q[50];
cx q[53],q[54];
rz(-pi) q[53];
x q[53];
cx q[50],q[53];
cx q[54],q[55];
rz(-pi) q[54];
x q[54];
cx q[53],q[54];
cx q[55],q[56];
rz(-pi) q[55];
x q[55];
cx q[54],q[55];
cx q[56],q[57];
rz(-pi) q[56];
x q[56];
cx q[55],q[56];
cx q[57],q[62];
rz(-pi) q[57];
x q[57];
cx q[56],q[57];
cx q[62],q[61];
cx q[61],q[60];
rz(-pi) q[61];
x q[61];
rz(-pi) q[62];
x q[62];
cx q[57],q[62];
cx q[62],q[61];
cx q[60],q[63];
rz(-pi) q[60];
x q[60];
cx q[61],q[60];
cx q[63],q[64];
rz(-pi) q[63];
x q[63];
cx q[60],q[63];
cx q[64],q[65];
rz(-pi) q[64];
x q[64];
cx q[63],q[64];
cx q[65],q[66];
rz(-pi) q[65];
x q[65];
cx q[64],q[65];
cx q[66],q[67];
rz(-pi) q[66];
x q[66];
cx q[65],q[66];
cx q[67],q[72];
rz(-pi) q[67];
x q[67];
cx q[66],q[67];
cx q[72],q[71];
cx q[71],q[70];
rz(-pi) q[71];
x q[71];
rz(-pi) q[72];
x q[72];
cx q[67],q[72];
cx q[72],q[71];
cx q[70],q[73];
rz(-pi) q[70];
x q[70];
cx q[71],q[70];
cx q[73],q[74];
rz(-pi) q[73];
x q[73];
cx q[70],q[73];
cx q[74],q[75];
rz(-pi) q[74];
x q[74];
cx q[73],q[74];
cx q[75],q[76];
rz(-pi) q[75];
x q[75];
cx q[74],q[75];
cx q[76],q[77];
rz(-pi) q[76];
x q[76];
cx q[75],q[76];
cx q[77],q[82];
rz(-pi) q[77];
x q[77];
cx q[76],q[77];
cx q[82],q[81];
cx q[81],q[80];
rz(-pi) q[81];
x q[81];
rz(-pi) q[82];
x q[82];
cx q[77],q[82];
cx q[82],q[81];
cx q[80],q[83];
rz(-pi) q[80];
x q[80];
cx q[81],q[80];
cx q[83],q[84];
rz(-pi) q[83];
x q[83];
cx q[80],q[83];
cx q[84],q[85];
rz(-pi) q[84];
x q[84];
cx q[83],q[84];
cx q[85],q[86];
rz(-pi) q[85];
x q[85];
cx q[84],q[85];
cx q[86],q[87];
rz(-pi) q[86];
x q[86];
cx q[85],q[86];
cx q[87],q[92];
rz(-pi) q[87];
x q[87];
cx q[86],q[87];
cx q[92],q[91];
cx q[91],q[90];
rz(-pi) q[91];
x q[91];
rz(-pi) q[92];
x q[92];
cx q[87],q[92];
cx q[92],q[91];
cx q[90],q[93];
rz(-pi) q[90];
x q[90];
cx q[91],q[90];
cx q[93],q[94];
rz(-pi) q[93];
x q[93];
cx q[90],q[93];
cx q[94],q[95];
rz(-pi) q[94];
x q[94];
cx q[93],q[94];
cx q[95],q[96];
rz(-pi) q[95];
x q[95];
cx q[94],q[95];
cx q[96],q[97];
rz(-pi) q[96];
x q[96];
cx q[95],q[96];
cx q[97],q[102];
cx q[102],q[101];
cx q[101],q[100];
rz(-pi) q[101];
x q[101];
rz(-pi) q[102];
x q[102];
rz(-pi) q[97];
x q[97];
cx q[96],q[97];
cx q[97],q[102];
cx q[102],q[101];
cx q[100],q[103];
rz(-pi) q[100];
x q[100];
cx q[101],q[100];
cx q[103],q[104];
rz(-pi) q[103];
x q[103];
cx q[100],q[103];
cx q[104],q[105];
rz(-pi) q[104];
x q[104];
cx q[103],q[104];
cx q[105],q[106];
rz(-pi) q[105];
x q[105];
cx q[104],q[105];
cx q[106],q[107];
rz(-pi) q[106];
x q[106];
cx q[105],q[106];
cx q[107],q[112];
rz(-pi) q[107];
x q[107];
cx q[106],q[107];
cx q[112],q[111];
cx q[111],q[110];
rz(-pi) q[111];
x q[111];
rz(-pi) q[112];
x q[112];
cx q[107],q[112];
cx q[112],q[111];
cx q[110],q[113];
rz(-pi) q[110];
x q[110];
cx q[111],q[110];
cx q[113],q[114];
rz(-pi) q[113];
x q[113];
cx q[110],q[113];
cx q[114],q[115];
rz(-pi) q[114];
x q[114];
cx q[113],q[114];
cx q[115],q[116];
rz(-pi) q[115];
x q[115];
cx q[114],q[115];
cx q[116],q[117];
rz(-pi) q[116];
x q[116];
cx q[115],q[116];
cx q[117],q[122];
rz(-pi) q[117];
x q[117];
cx q[116],q[117];
cx q[122],q[121];
cx q[121],q[120];
rz(-pi) q[121];
x q[121];
rz(-pi) q[122];
x q[122];
cx q[117],q[122];
cx q[122],q[121];
cx q[120],q[123];
rz(-pi) q[120];
x q[120];
cx q[121],q[120];
cx q[123],q[124];
rz(-pi) q[123];
x q[123];
cx q[120],q[123];
cx q[124],q[125];
rz(-pi) q[124];
x q[124];
cx q[123],q[124];
cx q[125],q[126];
rz(-pi) q[125];
x q[125];
cx q[124],q[125];
cx q[126],q[127];
rz(-pi) q[126];
x q[126];
cx q[125],q[126];
cx q[127],q[132];
rz(-pi) q[127];
x q[127];
cx q[126],q[127];
cx q[132],q[131];
cx q[131],q[130];
rz(-pi) q[131];
x q[131];
rz(-pi) q[132];
x q[132];
cx q[127],q[132];
cx q[132],q[131];
cx q[130],q[133];
rz(-pi) q[130];
x q[130];
cx q[131],q[130];
cx q[133],q[134];
rz(-pi) q[133];
x q[133];
cx q[130],q[133];
cx q[134],q[135];
rz(-pi) q[134];
x q[134];
cx q[133],q[134];
cx q[135],q[136];
rz(-pi) q[135];
x q[135];
cx q[134],q[135];
cx q[136],q[137];
rz(-pi) q[136];
x q[136];
cx q[135],q[136];
cx q[137],q[142];
rz(-pi) q[137];
x q[137];
cx q[136],q[137];
cx q[142],q[141];
cx q[141],q[140];
rz(-pi) q[141];
x q[141];
rz(-pi) q[142];
x q[142];
cx q[137],q[142];
cx q[142],q[141];
cx q[140],q[143];
rz(-pi) q[140];
x q[140];
cx q[141],q[140];
cx q[143],q[144];
rz(-pi) q[143];
x q[143];
cx q[140],q[143];
cx q[144],q[145];
rz(-pi) q[144];
x q[144];
cx q[143],q[144];
cx q[145],q[146];
rz(-pi) q[145];
x q[145];
cx q[144],q[145];
cx q[146],q[147];
rz(-pi) q[146];
x q[146];
cx q[145],q[146];
cx q[147],q[152];
rz(-pi) q[147];
x q[147];
cx q[146],q[147];
cx q[152],q[151];
cx q[151],q[150];
rz(-pi) q[151];
x q[151];
rz(-pi) q[152];
x q[152];
cx q[147],q[152];
cx q[152],q[151];
cx q[150],q[153];
rz(-pi) q[150];
x q[150];
cx q[151],q[150];
cx q[153],q[154];
rz(-pi) q[153];
x q[153];
cx q[150],q[153];
cx q[154],q[155];
cx q[153],q[154];
cx q[155],q[156];
cx q[154],q[155];
cx q[156],q[157];
cx q[155],q[156];
cx q[157],q[162];
cx q[156],q[157];
cx q[162],q[161];
cx q[157],q[162];
cx q[161],q[160];
cx q[162],q[161];
cx q[160],q[163];
cx q[161],q[160];
cx q[163],q[164];
cx q[160],q[163];
cx q[164],q[165];
cx q[163],q[164];
cx q[165],q[166];
cx q[164],q[165];
cx q[166],q[167];
cx q[165],q[166];
cx q[167],q[172];
cx q[166],q[167];
cx q[172],q[171];
cx q[167],q[172];
cx q[171],q[170];
cx q[172],q[171];
cx q[170],q[173];
cx q[171],q[170];
cx q[173],q[174];
cx q[170],q[173];
cx q[174],q[175];
cx q[173],q[174];
cx q[175],q[176];
cx q[174],q[175];
cx q[176],q[177];
cx q[175],q[176];
cx q[177],q[182];
cx q[176],q[177];
cx q[182],q[181];
cx q[177],q[182];
cx q[181],q[180];
cx q[182],q[181];
cx q[180],q[183];
cx q[181],q[180];
cx q[183],q[184];
cx q[180],q[183];
cx q[184],q[185];
cx q[183],q[184];
cx q[185],q[186];
cx q[184],q[185];
cx q[186],q[187];
cx q[185],q[186];
cx q[187],q[192];
cx q[186],q[187];
cx q[192],q[191];
cx q[187],q[192];
cx q[191],q[190];
cx q[192],q[191];
cx q[190],q[193];
cx q[191],q[190];
cx q[193],q[194];
cx q[190],q[193];
cx q[194],q[195];
cx q[193],q[194];
cx q[195],q[196];
cx q[194],q[195];
cx q[196],q[197];
cx q[195],q[196];
cx q[197],q[202];
cx q[196],q[197];
cx q[202],q[201];
cx q[197],q[202];
cx q[201],q[200];
cx q[202],q[201];
cx q[200],q[203];
cx q[201],q[200];
cx q[203],q[204];
cx q[200],q[203];
cx q[204],q[205];
cx q[203],q[204];
cx q[205],q[206];
cx q[204],q[205];
cx q[206],q[207];
cx q[205],q[206];
cx q[207],q[212];
cx q[206],q[207];
cx q[212],q[211];
cx q[207],q[212];
cx q[211],q[210];
cx q[212],q[211];
cx q[210],q[213];
cx q[211],q[210];
cx q[213],q[214];
cx q[210],q[213];
cx q[214],q[215];
cx q[213],q[214];
cx q[215],q[216];
cx q[214],q[215];
cx q[216],q[217];
cx q[215],q[216];
cx q[217],q[222];
cx q[216],q[217];
cx q[222],q[221];
cx q[217],q[222];
cx q[221],q[220];
cx q[222],q[221];
cx q[220],q[223];
cx q[221],q[220];
cx q[223],q[224];
cx q[220],q[223];
cx q[224],q[225];
cx q[223],q[224];
cx q[225],q[226];
cx q[224],q[225];
cx q[226],q[227];
cx q[225],q[226];
cx q[227],q[232];
cx q[226],q[227];
cx q[232],q[231];
cx q[227],q[232];
cx q[231],q[230];
cx q[232],q[231];
cx q[230],q[233];
cx q[231],q[230];
cx q[233],q[234];
cx q[230],q[233];
cx q[234],q[235];
cx q[233],q[234];
cx q[235],q[236];
cx q[234],q[235];
cx q[236],q[237];
cx q[235],q[236];
cx q[237],q[242];
cx q[236],q[237];
cx q[242],q[241];
cx q[237],q[242];
cx q[241],q[240];
cx q[242],q[241];
cx q[240],q[243];
cx q[241],q[240];
cx q[243],q[244];
cx q[240],q[243];
cx q[244],q[245];
cx q[243],q[244];
cx q[245],q[246];
cx q[244],q[245];
cx q[246],q[247];
cx q[245],q[246];
cx q[247],q[252];
cx q[246],q[247];
cx q[252],q[251];
cx q[247],q[252];
cx q[251],q[250];
cx q[252],q[251];
cx q[250],q[253];
cx q[251],q[250];
cx q[253],q[254];
cx q[250],q[253];
cx q[254],q[255];
cx q[253],q[254];
cx q[255],q[256];
cx q[254],q[255];
cx q[256],q[257];
cx q[255],q[256];
cx q[257],q[262];
cx q[256],q[257];
cx q[262],q[261];
cx q[257],q[262];
cx q[261],q[260];
cx q[262],q[261];
cx q[260],q[263];
cx q[261],q[260];
cx q[263],q[264];
cx q[260],q[263];
cx q[264],q[265];
cx q[263],q[264];
cx q[265],q[266];
cx q[264],q[265];
cx q[266],q[267];
cx q[265],q[266];
cx q[267],q[272];
cx q[266],q[267];
cx q[272],q[271];
cx q[267],q[272];
cx q[271],q[270];
cx q[272],q[271];
cx q[270],q[273];
cx q[271],q[270];
cx q[273],q[274];
cx q[270],q[273];
cx q[274],q[275];
cx q[273],q[274];
cx q[275],q[276];
cx q[274],q[275];
cx q[276],q[277];
cx q[275],q[276];
cx q[277],q[282];
cx q[276],q[277];
cx q[282],q[281];
cx q[277],q[282];
cx q[281],q[280];
cx q[282],q[281];
cx q[280],q[283];
cx q[281],q[280];
cx q[283],q[284];
cx q[280],q[283];
cx q[284],q[285];
cx q[283],q[284];
cx q[285],q[286];
cx q[284],q[285];
cx q[286],q[287];
cx q[285],q[286];
cx q[287],q[292];
cx q[286],q[287];
cx q[292],q[291];
cx q[287],q[292];
cx q[291],q[290];
cx q[292],q[291];
cx q[290],q[293];
cx q[291],q[290];
cx q[293],q[294];
cx q[290],q[293];
cx q[294],q[295];
cx q[293],q[294];
cx q[295],q[296];
cx q[294],q[295];
cx q[296],q[297];
cx q[295],q[296];
cx q[297],q[302];
cx q[296],q[297];
cx q[302],q[301];
cx q[297],q[302];
cx q[301],q[300];
cx q[302],q[301];
cx q[300],q[303];
cx q[301],q[300];
cx q[303],q[304];
cx q[300],q[303];
cx q[304],q[305];
cx q[303],q[304];
cx q[305],q[306];
cx q[304],q[305];
cx q[306],q[309];
cx q[305],q[306];
cx q[306],q[309];
