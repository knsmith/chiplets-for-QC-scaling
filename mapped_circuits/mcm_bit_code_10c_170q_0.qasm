OPENQASM 2.0;
include "qelib1.inc";
qreg q[170];
creg m_mcm0[67];
creg m_meas_all[135];
x q[1];
cx q[1],q[0];
x q[3];
cx q[3],q[0];
cx q[3],q[4];
x q[5];
cx q[5],q[4];
cx q[5],q[6];
x q[7];
cx q[7],q[6];
x q[11];
cx q[7],q[12];
cx q[11],q[12];
cx q[11],q[10];
x q[13];
cx q[13],q[10];
cx q[13],q[14];
x q[15];
cx q[15],q[14];
cx q[15],q[16];
x q[17];
cx q[17],q[16];
x q[21];
cx q[17],q[22];
cx q[21],q[22];
cx q[21],q[20];
x q[23];
cx q[23],q[20];
cx q[23],q[24];
x q[25];
cx q[25],q[24];
cx q[25],q[26];
x q[27];
cx q[27],q[26];
x q[31];
cx q[27],q[32];
cx q[31],q[32];
cx q[31],q[30];
x q[33];
cx q[33],q[30];
cx q[33],q[34];
x q[35];
cx q[35],q[34];
cx q[35],q[36];
x q[37];
cx q[37],q[36];
x q[41];
cx q[37],q[42];
cx q[41],q[42];
cx q[41],q[40];
x q[43];
cx q[43],q[40];
cx q[43],q[44];
x q[45];
cx q[45],q[44];
cx q[45],q[46];
x q[47];
cx q[47],q[46];
x q[51];
cx q[47],q[52];
cx q[51],q[52];
cx q[51],q[50];
x q[53];
cx q[53],q[50];
cx q[53],q[54];
x q[55];
cx q[55],q[54];
cx q[55],q[56];
x q[57];
cx q[57],q[56];
x q[61];
cx q[57],q[62];
cx q[61],q[62];
cx q[61],q[60];
x q[63];
cx q[63],q[60];
cx q[63],q[64];
x q[65];
cx q[65],q[64];
cx q[65],q[66];
x q[67];
cx q[67],q[66];
x q[71];
cx q[67],q[72];
cx q[71],q[72];
cx q[71],q[70];
x q[73];
cx q[73],q[70];
cx q[73],q[74];
x q[75];
cx q[75],q[74];
cx q[75],q[76];
x q[77];
cx q[77],q[76];
x q[81];
cx q[77],q[82];
cx q[81],q[82];
cx q[81],q[80];
x q[83];
cx q[83],q[80];
cx q[83],q[84];
x q[85];
cx q[85],q[84];
cx q[85],q[86];
x q[87];
cx q[87],q[86];
x q[91];
cx q[87],q[92];
cx q[91],q[92];
cx q[91],q[90];
x q[93];
cx q[93],q[90];
cx q[93],q[94];
x q[95];
cx q[95],q[94];
cx q[95],q[96];
x q[97];
cx q[97],q[96];
x q[101];
cx q[97],q[102];
cx q[101],q[102];
cx q[101],q[100];
x q[103];
cx q[103],q[100];
cx q[103],q[104];
x q[105];
cx q[105],q[104];
cx q[105],q[106];
x q[107];
cx q[107],q[106];
x q[111];
cx q[107],q[112];
cx q[111],q[112];
cx q[111],q[110];
x q[113];
cx q[113],q[110];
cx q[113],q[114];
x q[115];
cx q[115],q[114];
cx q[115],q[116];
x q[117];
cx q[117],q[116];
x q[121];
cx q[117],q[122];
cx q[121],q[122];
cx q[121],q[120];
x q[123];
cx q[123],q[120];
cx q[123],q[124];
x q[125];
cx q[125],q[124];
cx q[125],q[126];
x q[127];
cx q[127],q[126];
x q[131];
cx q[127],q[132];
cx q[131],q[132];
cx q[131],q[130];
x q[133];
cx q[133],q[130];
cx q[133],q[134];
x q[135];
cx q[135],q[134];
cx q[135],q[136];
x q[137];
cx q[137],q[136];
x q[141];
cx q[137],q[142];
cx q[141],q[142];
cx q[141],q[140];
x q[143];
cx q[143],q[140];
cx q[143],q[144];
x q[145];
cx q[145],q[144];
cx q[145],q[146];
x q[147];
cx q[147],q[146];
x q[151];
cx q[147],q[152];
cx q[151],q[152];
cx q[151],q[150];
x q[153];
cx q[153],q[150];
cx q[153],q[154];
x q[155];
cx q[155],q[154];
cx q[155],q[156];
x q[157];
cx q[157],q[156];
x q[161];
cx q[157],q[162];
cx q[161],q[162];
cx q[161],q[160];
x q[163];
cx q[163],q[160];
cx q[163],q[164];
x q[165];
cx q[165],q[164];
cx q[165],q[166];
x q[169];
cx q[169],q[166];
measure q[0] -> m_mcm0[0];
reset q[0];
measure q[4] -> m_mcm0[1];
reset q[4];
measure q[6] -> m_mcm0[2];
reset q[6];
measure q[12] -> m_mcm0[3];
reset q[12];
measure q[10] -> m_mcm0[4];
reset q[10];
measure q[14] -> m_mcm0[5];
reset q[14];
measure q[16] -> m_mcm0[6];
reset q[16];
measure q[22] -> m_mcm0[7];
reset q[22];
measure q[20] -> m_mcm0[8];
reset q[20];
measure q[24] -> m_mcm0[9];
reset q[24];
measure q[26] -> m_mcm0[10];
reset q[26];
measure q[32] -> m_mcm0[11];
reset q[32];
measure q[30] -> m_mcm0[12];
reset q[30];
measure q[34] -> m_mcm0[13];
reset q[34];
measure q[36] -> m_mcm0[14];
reset q[36];
measure q[42] -> m_mcm0[15];
reset q[42];
measure q[40] -> m_mcm0[16];
reset q[40];
measure q[44] -> m_mcm0[17];
reset q[44];
measure q[46] -> m_mcm0[18];
reset q[46];
measure q[52] -> m_mcm0[19];
reset q[52];
measure q[50] -> m_mcm0[20];
reset q[50];
measure q[54] -> m_mcm0[21];
reset q[54];
measure q[56] -> m_mcm0[22];
reset q[56];
measure q[62] -> m_mcm0[23];
reset q[62];
measure q[60] -> m_mcm0[24];
reset q[60];
measure q[64] -> m_mcm0[25];
reset q[64];
measure q[66] -> m_mcm0[26];
reset q[66];
measure q[72] -> m_mcm0[27];
reset q[72];
measure q[70] -> m_mcm0[28];
reset q[70];
measure q[74] -> m_mcm0[29];
reset q[74];
measure q[76] -> m_mcm0[30];
reset q[76];
measure q[82] -> m_mcm0[31];
reset q[82];
measure q[80] -> m_mcm0[32];
reset q[80];
measure q[84] -> m_mcm0[33];
reset q[84];
measure q[86] -> m_mcm0[34];
reset q[86];
measure q[92] -> m_mcm0[35];
reset q[92];
measure q[90] -> m_mcm0[36];
reset q[90];
measure q[94] -> m_mcm0[37];
reset q[94];
measure q[96] -> m_mcm0[38];
reset q[96];
measure q[102] -> m_mcm0[39];
reset q[102];
measure q[100] -> m_mcm0[40];
reset q[100];
measure q[104] -> m_mcm0[41];
reset q[104];
measure q[106] -> m_mcm0[42];
reset q[106];
measure q[112] -> m_mcm0[43];
reset q[112];
measure q[110] -> m_mcm0[44];
reset q[110];
measure q[114] -> m_mcm0[45];
reset q[114];
measure q[116] -> m_mcm0[46];
reset q[116];
measure q[122] -> m_mcm0[47];
reset q[122];
measure q[120] -> m_mcm0[48];
reset q[120];
measure q[124] -> m_mcm0[49];
reset q[124];
measure q[126] -> m_mcm0[50];
reset q[126];
measure q[132] -> m_mcm0[51];
reset q[132];
measure q[130] -> m_mcm0[52];
reset q[130];
measure q[134] -> m_mcm0[53];
reset q[134];
measure q[136] -> m_mcm0[54];
reset q[136];
measure q[142] -> m_mcm0[55];
reset q[142];
measure q[140] -> m_mcm0[56];
reset q[140];
measure q[144] -> m_mcm0[57];
reset q[144];
measure q[146] -> m_mcm0[58];
reset q[146];
measure q[152] -> m_mcm0[59];
reset q[152];
measure q[150] -> m_mcm0[60];
reset q[150];
measure q[154] -> m_mcm0[61];
reset q[154];
measure q[156] -> m_mcm0[62];
reset q[156];
measure q[162] -> m_mcm0[63];
reset q[162];
measure q[160] -> m_mcm0[64];
reset q[160];
measure q[164] -> m_mcm0[65];
reset q[164];
measure q[166] -> m_mcm0[66];
reset q[166];
measure q[1] -> m_meas_all[0];
measure q[0] -> m_meas_all[1];
measure q[3] -> m_meas_all[2];
measure q[4] -> m_meas_all[3];
measure q[5] -> m_meas_all[4];
measure q[6] -> m_meas_all[5];
measure q[7] -> m_meas_all[6];
measure q[12] -> m_meas_all[7];
measure q[11] -> m_meas_all[8];
measure q[10] -> m_meas_all[9];
measure q[13] -> m_meas_all[10];
measure q[14] -> m_meas_all[11];
measure q[15] -> m_meas_all[12];
measure q[16] -> m_meas_all[13];
measure q[17] -> m_meas_all[14];
measure q[22] -> m_meas_all[15];
measure q[21] -> m_meas_all[16];
measure q[20] -> m_meas_all[17];
measure q[23] -> m_meas_all[18];
measure q[24] -> m_meas_all[19];
measure q[25] -> m_meas_all[20];
measure q[26] -> m_meas_all[21];
measure q[27] -> m_meas_all[22];
measure q[32] -> m_meas_all[23];
measure q[31] -> m_meas_all[24];
measure q[30] -> m_meas_all[25];
measure q[33] -> m_meas_all[26];
measure q[34] -> m_meas_all[27];
measure q[35] -> m_meas_all[28];
measure q[36] -> m_meas_all[29];
measure q[37] -> m_meas_all[30];
measure q[42] -> m_meas_all[31];
measure q[41] -> m_meas_all[32];
measure q[40] -> m_meas_all[33];
measure q[43] -> m_meas_all[34];
measure q[44] -> m_meas_all[35];
measure q[45] -> m_meas_all[36];
measure q[46] -> m_meas_all[37];
measure q[47] -> m_meas_all[38];
measure q[52] -> m_meas_all[39];
measure q[51] -> m_meas_all[40];
measure q[50] -> m_meas_all[41];
measure q[53] -> m_meas_all[42];
measure q[54] -> m_meas_all[43];
measure q[55] -> m_meas_all[44];
measure q[56] -> m_meas_all[45];
measure q[57] -> m_meas_all[46];
measure q[62] -> m_meas_all[47];
measure q[61] -> m_meas_all[48];
measure q[60] -> m_meas_all[49];
measure q[63] -> m_meas_all[50];
measure q[64] -> m_meas_all[51];
measure q[65] -> m_meas_all[52];
measure q[66] -> m_meas_all[53];
measure q[67] -> m_meas_all[54];
measure q[72] -> m_meas_all[55];
measure q[71] -> m_meas_all[56];
measure q[70] -> m_meas_all[57];
measure q[73] -> m_meas_all[58];
measure q[74] -> m_meas_all[59];
measure q[75] -> m_meas_all[60];
measure q[76] -> m_meas_all[61];
measure q[77] -> m_meas_all[62];
measure q[82] -> m_meas_all[63];
measure q[81] -> m_meas_all[64];
measure q[80] -> m_meas_all[65];
measure q[83] -> m_meas_all[66];
measure q[84] -> m_meas_all[67];
measure q[85] -> m_meas_all[68];
measure q[86] -> m_meas_all[69];
measure q[87] -> m_meas_all[70];
measure q[92] -> m_meas_all[71];
measure q[91] -> m_meas_all[72];
measure q[90] -> m_meas_all[73];
measure q[93] -> m_meas_all[74];
measure q[94] -> m_meas_all[75];
measure q[95] -> m_meas_all[76];
measure q[96] -> m_meas_all[77];
measure q[97] -> m_meas_all[78];
measure q[102] -> m_meas_all[79];
measure q[101] -> m_meas_all[80];
measure q[100] -> m_meas_all[81];
measure q[103] -> m_meas_all[82];
measure q[104] -> m_meas_all[83];
measure q[105] -> m_meas_all[84];
measure q[106] -> m_meas_all[85];
measure q[107] -> m_meas_all[86];
measure q[112] -> m_meas_all[87];
measure q[111] -> m_meas_all[88];
measure q[110] -> m_meas_all[89];
measure q[113] -> m_meas_all[90];
measure q[114] -> m_meas_all[91];
measure q[115] -> m_meas_all[92];
measure q[116] -> m_meas_all[93];
measure q[117] -> m_meas_all[94];
measure q[122] -> m_meas_all[95];
measure q[121] -> m_meas_all[96];
measure q[120] -> m_meas_all[97];
measure q[123] -> m_meas_all[98];
measure q[124] -> m_meas_all[99];
measure q[125] -> m_meas_all[100];
measure q[126] -> m_meas_all[101];
measure q[127] -> m_meas_all[102];
measure q[132] -> m_meas_all[103];
measure q[131] -> m_meas_all[104];
measure q[130] -> m_meas_all[105];
measure q[133] -> m_meas_all[106];
measure q[134] -> m_meas_all[107];
measure q[135] -> m_meas_all[108];
measure q[136] -> m_meas_all[109];
measure q[137] -> m_meas_all[110];
measure q[142] -> m_meas_all[111];
measure q[141] -> m_meas_all[112];
measure q[140] -> m_meas_all[113];
measure q[143] -> m_meas_all[114];
measure q[144] -> m_meas_all[115];
measure q[145] -> m_meas_all[116];
measure q[146] -> m_meas_all[117];
measure q[147] -> m_meas_all[118];
measure q[152] -> m_meas_all[119];
measure q[151] -> m_meas_all[120];
measure q[150] -> m_meas_all[121];
measure q[153] -> m_meas_all[122];
measure q[154] -> m_meas_all[123];
measure q[155] -> m_meas_all[124];
measure q[156] -> m_meas_all[125];
measure q[157] -> m_meas_all[126];
measure q[162] -> m_meas_all[127];
measure q[161] -> m_meas_all[128];
measure q[160] -> m_meas_all[129];
measure q[163] -> m_meas_all[130];
measure q[164] -> m_meas_all[131];
measure q[165] -> m_meas_all[132];
measure q[166] -> m_meas_all[133];
measure q[169] -> m_meas_all[134];
