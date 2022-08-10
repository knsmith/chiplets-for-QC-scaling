OPENQASM 2.0;
include "qelib1.inc";
qreg q[220];
creg m_mcm0[87];
creg m_meas_all[175];
x q[1];
x q[3];
x q[7];
x q[8];
x q[10];
x q[14];
x q[16];
x q[17];
x q[21];
x q[25];
x q[27];
x q[28];
x q[30];
x q[34];
x q[36];
x q[37];
x q[41];
x q[45];
x q[47];
x q[48];
x q[50];
x q[54];
x q[56];
x q[57];
x q[61];
x q[65];
x q[67];
x q[68];
x q[70];
x q[74];
x q[76];
x q[77];
x q[81];
x q[85];
x q[87];
x q[88];
x q[90];
x q[94];
x q[96];
x q[97];
x q[101];
x q[105];
x q[107];
x q[108];
x q[110];
x q[114];
x q[116];
x q[117];
x q[121];
x q[125];
x q[127];
x q[128];
x q[130];
x q[134];
x q[136];
x q[137];
x q[141];
x q[143];
cx q[143],q[144];
x q[145];
cx q[145],q[144];
cx q[145],q[146];
cx q[137],q[146];
cx q[137],q[135];
cx q[134],q[135];
cx q[134],q[133];
cx q[128],q[133];
cx q[128],q[124];
cx q[125],q[124];
cx q[125],q[126];
cx q[117],q[126];
cx q[117],q[115];
cx q[114],q[115];
cx q[114],q[113];
cx q[108],q[113];
cx q[108],q[104];
cx q[105],q[104];
cx q[105],q[106];
cx q[97],q[106];
cx q[97],q[95];
cx q[94],q[95];
cx q[94],q[93];
cx q[88],q[93];
cx q[88],q[84];
cx q[85],q[84];
cx q[85],q[86];
cx q[77],q[86];
cx q[77],q[75];
cx q[74],q[75];
cx q[74],q[73];
cx q[68],q[73];
cx q[68],q[64];
cx q[65],q[64];
cx q[65],q[66];
cx q[57],q[66];
cx q[57],q[55];
cx q[54],q[55];
cx q[54],q[53];
cx q[48],q[53];
cx q[48],q[44];
cx q[45],q[44];
cx q[45],q[46];
cx q[37],q[46];
cx q[37],q[35];
cx q[34],q[35];
cx q[34],q[33];
cx q[28],q[33];
cx q[28],q[24];
cx q[25],q[24];
cx q[25],q[26];
cx q[17],q[26];
cx q[17],q[15];
cx q[14],q[15];
cx q[14],q[13];
cx q[8],q[13];
cx q[8],q[4];
cx q[3],q[4];
cx q[3],q[2];
cx q[1],q[2];
cx q[1],q[0];
cx q[7],q[0];
cx q[7],q[9];
cx q[10],q[9];
cx q[10],q[11];
cx q[16],q[11];
cx q[16],q[22];
cx q[21],q[22];
cx q[21],q[20];
cx q[27],q[20];
cx q[27],q[29];
cx q[30],q[29];
cx q[30],q[31];
cx q[36],q[31];
cx q[36],q[42];
cx q[41],q[42];
cx q[41],q[40];
cx q[47],q[40];
cx q[47],q[49];
cx q[50],q[49];
cx q[50],q[51];
cx q[56],q[51];
cx q[56],q[62];
cx q[61],q[62];
cx q[61],q[60];
cx q[67],q[60];
cx q[67],q[69];
cx q[70],q[69];
cx q[70],q[71];
cx q[76],q[71];
cx q[76],q[82];
cx q[81],q[82];
cx q[81],q[80];
cx q[87],q[80];
cx q[87],q[89];
cx q[90],q[89];
cx q[90],q[91];
cx q[96],q[91];
cx q[96],q[102];
cx q[101],q[102];
cx q[101],q[100];
cx q[107],q[100];
cx q[107],q[109];
cx q[110],q[109];
cx q[110],q[111];
cx q[116],q[111];
cx q[116],q[122];
cx q[121],q[122];
cx q[121],q[120];
cx q[127],q[120];
cx q[127],q[129];
cx q[130],q[129];
cx q[130],q[131];
cx q[136],q[131];
cx q[136],q[142];
cx q[141],q[142];
cx q[141],q[140];
x q[147];
cx q[147],q[140];
cx q[147],q[149];
x q[150];
cx q[150],q[149];
cx q[150],q[151];
x q[152];
cx q[152],q[151];
cx q[152],q[153];
x q[154];
cx q[154],q[153];
cx q[154],q[155];
x q[157];
cx q[157],q[155];
x q[161];
x q[163];
x q[165];
cx q[157],q[166];
cx q[165],q[166];
cx q[165],q[164];
cx q[163],q[164];
cx q[163],q[162];
cx q[161],q[162];
cx q[161],q[160];
x q[167];
cx q[167],q[160];
cx q[167],q[169];
x q[170];
cx q[170],q[169];
cx q[170],q[171];
x q[172];
cx q[172],q[171];
cx q[172],q[173];
x q[174];
cx q[174],q[173];
cx q[174],q[175];
x q[177];
cx q[177],q[175];
x q[181];
x q[183];
x q[185];
cx q[177],q[186];
cx q[185],q[186];
cx q[185],q[184];
cx q[183],q[184];
cx q[183],q[182];
cx q[181],q[182];
cx q[181],q[180];
x q[187];
cx q[187],q[180];
cx q[187],q[189];
x q[190];
cx q[190],q[189];
cx q[190],q[191];
x q[192];
cx q[192],q[191];
cx q[192],q[193];
x q[194];
cx q[194],q[193];
cx q[194],q[195];
x q[197];
cx q[197],q[195];
x q[201];
x q[203];
x q[205];
cx q[197],q[206];
cx q[205],q[206];
cx q[205],q[204];
cx q[203],q[204];
cx q[203],q[202];
cx q[201],q[202];
cx q[201],q[200];
x q[207];
cx q[207],q[200];
cx q[207],q[209];
x q[210];
cx q[210],q[209];
cx q[210],q[211];
x q[212];
cx q[212],q[211];
cx q[212],q[213];
x q[214];
cx q[214],q[213];
cx q[214],q[215];
x q[219];
cx q[219],q[215];
measure q[144] -> m_mcm0[0];
reset q[144];
measure q[146] -> m_mcm0[1];
reset q[146];
measure q[135] -> m_mcm0[2];
reset q[135];
measure q[133] -> m_mcm0[3];
reset q[133];
measure q[124] -> m_mcm0[4];
reset q[124];
measure q[126] -> m_mcm0[5];
reset q[126];
measure q[115] -> m_mcm0[6];
reset q[115];
measure q[113] -> m_mcm0[7];
reset q[113];
measure q[104] -> m_mcm0[8];
reset q[104];
measure q[106] -> m_mcm0[9];
reset q[106];
measure q[95] -> m_mcm0[10];
reset q[95];
measure q[93] -> m_mcm0[11];
reset q[93];
measure q[84] -> m_mcm0[12];
reset q[84];
measure q[86] -> m_mcm0[13];
reset q[86];
measure q[75] -> m_mcm0[14];
reset q[75];
measure q[73] -> m_mcm0[15];
reset q[73];
measure q[64] -> m_mcm0[16];
reset q[64];
measure q[66] -> m_mcm0[17];
reset q[66];
measure q[55] -> m_mcm0[18];
reset q[55];
measure q[53] -> m_mcm0[19];
reset q[53];
measure q[44] -> m_mcm0[20];
reset q[44];
measure q[46] -> m_mcm0[21];
reset q[46];
measure q[35] -> m_mcm0[22];
reset q[35];
measure q[33] -> m_mcm0[23];
reset q[33];
measure q[24] -> m_mcm0[24];
reset q[24];
measure q[26] -> m_mcm0[25];
reset q[26];
measure q[15] -> m_mcm0[26];
reset q[15];
measure q[13] -> m_mcm0[27];
reset q[13];
measure q[4] -> m_mcm0[28];
reset q[4];
measure q[2] -> m_mcm0[29];
reset q[2];
measure q[0] -> m_mcm0[30];
reset q[0];
measure q[9] -> m_mcm0[31];
reset q[9];
measure q[11] -> m_mcm0[32];
reset q[11];
measure q[22] -> m_mcm0[33];
reset q[22];
measure q[20] -> m_mcm0[34];
reset q[20];
measure q[29] -> m_mcm0[35];
reset q[29];
measure q[31] -> m_mcm0[36];
reset q[31];
measure q[42] -> m_mcm0[37];
reset q[42];
measure q[40] -> m_mcm0[38];
reset q[40];
measure q[49] -> m_mcm0[39];
reset q[49];
measure q[51] -> m_mcm0[40];
reset q[51];
measure q[62] -> m_mcm0[41];
reset q[62];
measure q[60] -> m_mcm0[42];
reset q[60];
measure q[69] -> m_mcm0[43];
reset q[69];
measure q[71] -> m_mcm0[44];
reset q[71];
measure q[82] -> m_mcm0[45];
reset q[82];
measure q[80] -> m_mcm0[46];
reset q[80];
measure q[89] -> m_mcm0[47];
reset q[89];
measure q[91] -> m_mcm0[48];
reset q[91];
measure q[102] -> m_mcm0[49];
reset q[102];
measure q[100] -> m_mcm0[50];
reset q[100];
measure q[109] -> m_mcm0[51];
reset q[109];
measure q[111] -> m_mcm0[52];
reset q[111];
measure q[122] -> m_mcm0[53];
reset q[122];
measure q[120] -> m_mcm0[54];
reset q[120];
measure q[129] -> m_mcm0[55];
reset q[129];
measure q[131] -> m_mcm0[56];
reset q[131];
measure q[142] -> m_mcm0[57];
reset q[142];
measure q[140] -> m_mcm0[58];
reset q[140];
measure q[149] -> m_mcm0[59];
reset q[149];
measure q[151] -> m_mcm0[60];
reset q[151];
measure q[153] -> m_mcm0[61];
reset q[153];
measure q[155] -> m_mcm0[62];
reset q[155];
measure q[166] -> m_mcm0[63];
reset q[166];
measure q[164] -> m_mcm0[64];
reset q[164];
measure q[162] -> m_mcm0[65];
reset q[162];
measure q[160] -> m_mcm0[66];
reset q[160];
measure q[169] -> m_mcm0[67];
reset q[169];
measure q[171] -> m_mcm0[68];
reset q[171];
measure q[173] -> m_mcm0[69];
reset q[173];
measure q[175] -> m_mcm0[70];
reset q[175];
measure q[186] -> m_mcm0[71];
reset q[186];
measure q[184] -> m_mcm0[72];
reset q[184];
measure q[182] -> m_mcm0[73];
reset q[182];
measure q[180] -> m_mcm0[74];
reset q[180];
measure q[189] -> m_mcm0[75];
reset q[189];
measure q[191] -> m_mcm0[76];
reset q[191];
measure q[193] -> m_mcm0[77];
reset q[193];
measure q[195] -> m_mcm0[78];
reset q[195];
measure q[206] -> m_mcm0[79];
reset q[206];
measure q[204] -> m_mcm0[80];
reset q[204];
measure q[202] -> m_mcm0[81];
reset q[202];
measure q[200] -> m_mcm0[82];
reset q[200];
measure q[209] -> m_mcm0[83];
reset q[209];
measure q[211] -> m_mcm0[84];
reset q[211];
measure q[213] -> m_mcm0[85];
reset q[213];
measure q[215] -> m_mcm0[86];
reset q[215];
measure q[143] -> m_meas_all[0];
measure q[144] -> m_meas_all[1];
measure q[145] -> m_meas_all[2];
measure q[146] -> m_meas_all[3];
measure q[137] -> m_meas_all[4];
measure q[135] -> m_meas_all[5];
measure q[134] -> m_meas_all[6];
measure q[133] -> m_meas_all[7];
measure q[128] -> m_meas_all[8];
measure q[124] -> m_meas_all[9];
measure q[125] -> m_meas_all[10];
measure q[126] -> m_meas_all[11];
measure q[117] -> m_meas_all[12];
measure q[115] -> m_meas_all[13];
measure q[114] -> m_meas_all[14];
measure q[113] -> m_meas_all[15];
measure q[108] -> m_meas_all[16];
measure q[104] -> m_meas_all[17];
measure q[105] -> m_meas_all[18];
measure q[106] -> m_meas_all[19];
measure q[97] -> m_meas_all[20];
measure q[95] -> m_meas_all[21];
measure q[94] -> m_meas_all[22];
measure q[93] -> m_meas_all[23];
measure q[88] -> m_meas_all[24];
measure q[84] -> m_meas_all[25];
measure q[85] -> m_meas_all[26];
measure q[86] -> m_meas_all[27];
measure q[77] -> m_meas_all[28];
measure q[75] -> m_meas_all[29];
measure q[74] -> m_meas_all[30];
measure q[73] -> m_meas_all[31];
measure q[68] -> m_meas_all[32];
measure q[64] -> m_meas_all[33];
measure q[65] -> m_meas_all[34];
measure q[66] -> m_meas_all[35];
measure q[57] -> m_meas_all[36];
measure q[55] -> m_meas_all[37];
measure q[54] -> m_meas_all[38];
measure q[53] -> m_meas_all[39];
measure q[48] -> m_meas_all[40];
measure q[44] -> m_meas_all[41];
measure q[45] -> m_meas_all[42];
measure q[46] -> m_meas_all[43];
measure q[37] -> m_meas_all[44];
measure q[35] -> m_meas_all[45];
measure q[34] -> m_meas_all[46];
measure q[33] -> m_meas_all[47];
measure q[28] -> m_meas_all[48];
measure q[24] -> m_meas_all[49];
measure q[25] -> m_meas_all[50];
measure q[26] -> m_meas_all[51];
measure q[17] -> m_meas_all[52];
measure q[15] -> m_meas_all[53];
measure q[14] -> m_meas_all[54];
measure q[13] -> m_meas_all[55];
measure q[8] -> m_meas_all[56];
measure q[4] -> m_meas_all[57];
measure q[3] -> m_meas_all[58];
measure q[2] -> m_meas_all[59];
measure q[1] -> m_meas_all[60];
measure q[0] -> m_meas_all[61];
measure q[7] -> m_meas_all[62];
measure q[9] -> m_meas_all[63];
measure q[10] -> m_meas_all[64];
measure q[11] -> m_meas_all[65];
measure q[16] -> m_meas_all[66];
measure q[22] -> m_meas_all[67];
measure q[21] -> m_meas_all[68];
measure q[20] -> m_meas_all[69];
measure q[27] -> m_meas_all[70];
measure q[29] -> m_meas_all[71];
measure q[30] -> m_meas_all[72];
measure q[31] -> m_meas_all[73];
measure q[36] -> m_meas_all[74];
measure q[42] -> m_meas_all[75];
measure q[41] -> m_meas_all[76];
measure q[40] -> m_meas_all[77];
measure q[47] -> m_meas_all[78];
measure q[49] -> m_meas_all[79];
measure q[50] -> m_meas_all[80];
measure q[51] -> m_meas_all[81];
measure q[56] -> m_meas_all[82];
measure q[62] -> m_meas_all[83];
measure q[61] -> m_meas_all[84];
measure q[60] -> m_meas_all[85];
measure q[67] -> m_meas_all[86];
measure q[69] -> m_meas_all[87];
measure q[70] -> m_meas_all[88];
measure q[71] -> m_meas_all[89];
measure q[76] -> m_meas_all[90];
measure q[82] -> m_meas_all[91];
measure q[81] -> m_meas_all[92];
measure q[80] -> m_meas_all[93];
measure q[87] -> m_meas_all[94];
measure q[89] -> m_meas_all[95];
measure q[90] -> m_meas_all[96];
measure q[91] -> m_meas_all[97];
measure q[96] -> m_meas_all[98];
measure q[102] -> m_meas_all[99];
measure q[101] -> m_meas_all[100];
measure q[100] -> m_meas_all[101];
measure q[107] -> m_meas_all[102];
measure q[109] -> m_meas_all[103];
measure q[110] -> m_meas_all[104];
measure q[111] -> m_meas_all[105];
measure q[116] -> m_meas_all[106];
measure q[122] -> m_meas_all[107];
measure q[121] -> m_meas_all[108];
measure q[120] -> m_meas_all[109];
measure q[127] -> m_meas_all[110];
measure q[129] -> m_meas_all[111];
measure q[130] -> m_meas_all[112];
measure q[131] -> m_meas_all[113];
measure q[136] -> m_meas_all[114];
measure q[142] -> m_meas_all[115];
measure q[141] -> m_meas_all[116];
measure q[140] -> m_meas_all[117];
measure q[147] -> m_meas_all[118];
measure q[149] -> m_meas_all[119];
measure q[150] -> m_meas_all[120];
measure q[151] -> m_meas_all[121];
measure q[152] -> m_meas_all[122];
measure q[153] -> m_meas_all[123];
measure q[154] -> m_meas_all[124];
measure q[155] -> m_meas_all[125];
measure q[157] -> m_meas_all[126];
measure q[166] -> m_meas_all[127];
measure q[165] -> m_meas_all[128];
measure q[164] -> m_meas_all[129];
measure q[163] -> m_meas_all[130];
measure q[162] -> m_meas_all[131];
measure q[161] -> m_meas_all[132];
measure q[160] -> m_meas_all[133];
measure q[167] -> m_meas_all[134];
measure q[169] -> m_meas_all[135];
measure q[170] -> m_meas_all[136];
measure q[171] -> m_meas_all[137];
measure q[172] -> m_meas_all[138];
measure q[173] -> m_meas_all[139];
measure q[174] -> m_meas_all[140];
measure q[175] -> m_meas_all[141];
measure q[177] -> m_meas_all[142];
measure q[186] -> m_meas_all[143];
measure q[185] -> m_meas_all[144];
measure q[184] -> m_meas_all[145];
measure q[183] -> m_meas_all[146];
measure q[182] -> m_meas_all[147];
measure q[181] -> m_meas_all[148];
measure q[180] -> m_meas_all[149];
measure q[187] -> m_meas_all[150];
measure q[189] -> m_meas_all[151];
measure q[190] -> m_meas_all[152];
measure q[191] -> m_meas_all[153];
measure q[192] -> m_meas_all[154];
measure q[193] -> m_meas_all[155];
measure q[194] -> m_meas_all[156];
measure q[195] -> m_meas_all[157];
measure q[197] -> m_meas_all[158];
measure q[206] -> m_meas_all[159];
measure q[205] -> m_meas_all[160];
measure q[204] -> m_meas_all[161];
measure q[203] -> m_meas_all[162];
measure q[202] -> m_meas_all[163];
measure q[201] -> m_meas_all[164];
measure q[200] -> m_meas_all[165];
measure q[207] -> m_meas_all[166];
measure q[209] -> m_meas_all[167];
measure q[210] -> m_meas_all[168];
measure q[211] -> m_meas_all[169];
measure q[212] -> m_meas_all[170];
measure q[213] -> m_meas_all[171];
measure q[214] -> m_meas_all[172];
measure q[215] -> m_meas_all[173];
measure q[219] -> m_meas_all[174];
