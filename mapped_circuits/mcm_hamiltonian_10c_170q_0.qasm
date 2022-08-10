OPENQASM 2.0;
include "qelib1.inc";
qreg q[170];
creg m0[136];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/4) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi/4) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/4) q[2];
sx q[2];
rz(-pi/2) q[2];
cx q[2],q[1];
rz(-pi/2) q[1];
cx q[2],q[1];
cx q[1],q[0];
rz(-pi/2) q[0];
cx q[1],q[0];
rz(-pi/2) q[3];
sx q[3];
rz(-pi/4) q[3];
sx q[3];
rz(-pi/2) q[3];
cx q[0],q[3];
rz(-pi/2) q[3];
cx q[0],q[3];
rz(-pi/2) q[4];
sx q[4];
rz(-pi/4) q[4];
sx q[4];
rz(-pi/2) q[4];
cx q[3],q[4];
rz(-pi/2) q[4];
cx q[3],q[4];
rz(-pi/2) q[5];
sx q[5];
rz(-pi/4) q[5];
sx q[5];
rz(-pi/2) q[5];
cx q[4],q[5];
rz(-pi/2) q[5];
cx q[4],q[5];
rz(-pi/2) q[6];
sx q[6];
rz(-pi/4) q[6];
sx q[6];
rz(-pi/2) q[6];
cx q[5],q[6];
rz(-pi/2) q[6];
cx q[5],q[6];
rz(-pi/2) q[7];
sx q[7];
rz(-pi/4) q[7];
sx q[7];
rz(-pi/2) q[7];
cx q[6],q[7];
rz(-pi/2) q[7];
cx q[6],q[7];
rz(-pi/2) q[10];
sx q[10];
rz(-pi/4) q[10];
sx q[10];
rz(-pi/2) q[10];
rz(-pi/2) q[11];
sx q[11];
rz(-pi/4) q[11];
sx q[11];
rz(-pi/2) q[11];
rz(-pi/2) q[12];
sx q[12];
rz(-pi/4) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[7],q[12];
rz(-pi/2) q[12];
cx q[7],q[12];
cx q[12],q[11];
rz(-pi/2) q[11];
cx q[12],q[11];
cx q[11],q[10];
rz(-pi/2) q[10];
cx q[11],q[10];
rz(-pi/2) q[13];
sx q[13];
rz(-pi/4) q[13];
sx q[13];
rz(-pi/2) q[13];
cx q[10],q[13];
rz(-pi/2) q[13];
cx q[10],q[13];
rz(-pi/2) q[14];
sx q[14];
rz(-pi/4) q[14];
sx q[14];
rz(-pi/2) q[14];
cx q[13],q[14];
rz(-pi/2) q[14];
cx q[13],q[14];
rz(-pi/2) q[15];
sx q[15];
rz(-pi/4) q[15];
sx q[15];
rz(-pi/2) q[15];
cx q[14],q[15];
rz(-pi/2) q[15];
cx q[14],q[15];
rz(-pi/2) q[16];
sx q[16];
rz(-pi/4) q[16];
sx q[16];
rz(-pi/2) q[16];
cx q[15],q[16];
rz(-pi/2) q[16];
cx q[15],q[16];
rz(-pi/2) q[17];
sx q[17];
rz(-pi/4) q[17];
sx q[17];
rz(-pi/2) q[17];
cx q[16],q[17];
rz(-pi/2) q[17];
cx q[16],q[17];
rz(-pi/2) q[20];
sx q[20];
rz(-pi/4) q[20];
sx q[20];
rz(-pi/2) q[20];
rz(-pi/2) q[21];
sx q[21];
rz(-pi/4) q[21];
sx q[21];
rz(-pi/2) q[21];
rz(-pi/2) q[22];
sx q[22];
rz(-pi/4) q[22];
sx q[22];
rz(-pi/2) q[22];
cx q[17],q[22];
rz(-pi/2) q[22];
cx q[17],q[22];
cx q[22],q[21];
rz(-pi/2) q[21];
cx q[22],q[21];
cx q[21],q[20];
rz(-pi/2) q[20];
cx q[21],q[20];
rz(-pi/2) q[23];
sx q[23];
rz(-pi/4) q[23];
sx q[23];
rz(-pi/2) q[23];
cx q[20],q[23];
rz(-pi/2) q[23];
cx q[20],q[23];
rz(-pi/2) q[24];
sx q[24];
rz(-pi/4) q[24];
sx q[24];
rz(-pi/2) q[24];
cx q[23],q[24];
rz(-pi/2) q[24];
cx q[23],q[24];
rz(-pi/2) q[25];
sx q[25];
rz(-pi/4) q[25];
sx q[25];
rz(-pi/2) q[25];
cx q[24],q[25];
rz(-pi/2) q[25];
cx q[24],q[25];
rz(-pi/2) q[26];
sx q[26];
rz(-pi/4) q[26];
sx q[26];
rz(-pi/2) q[26];
cx q[25],q[26];
rz(-pi/2) q[26];
cx q[25],q[26];
rz(-pi/2) q[27];
sx q[27];
rz(-pi/4) q[27];
sx q[27];
rz(-pi/2) q[27];
cx q[26],q[27];
rz(-pi/2) q[27];
cx q[26],q[27];
rz(-pi/2) q[30];
sx q[30];
rz(-pi/4) q[30];
sx q[30];
rz(-pi/2) q[30];
rz(-pi/2) q[31];
sx q[31];
rz(-pi/4) q[31];
sx q[31];
rz(-pi/2) q[31];
rz(-pi/2) q[32];
sx q[32];
rz(-pi/4) q[32];
sx q[32];
rz(-pi/2) q[32];
cx q[27],q[32];
rz(-pi/2) q[32];
cx q[27],q[32];
cx q[32],q[31];
rz(-pi/2) q[31];
cx q[32],q[31];
cx q[31],q[30];
rz(-pi/2) q[30];
cx q[31],q[30];
rz(-pi/2) q[33];
sx q[33];
rz(-pi/4) q[33];
sx q[33];
rz(-pi/2) q[33];
cx q[30],q[33];
rz(-pi/2) q[33];
cx q[30],q[33];
rz(-pi/2) q[34];
sx q[34];
rz(-pi/4) q[34];
sx q[34];
rz(-pi/2) q[34];
cx q[33],q[34];
rz(-pi/2) q[34];
cx q[33],q[34];
rz(-pi/2) q[35];
sx q[35];
rz(-pi/4) q[35];
sx q[35];
rz(-pi/2) q[35];
cx q[34],q[35];
rz(-pi/2) q[35];
cx q[34],q[35];
rz(-pi/2) q[36];
sx q[36];
rz(-pi/4) q[36];
sx q[36];
rz(-pi/2) q[36];
cx q[35],q[36];
rz(-pi/2) q[36];
cx q[35],q[36];
rz(-pi/2) q[37];
sx q[37];
rz(-pi/4) q[37];
sx q[37];
rz(-pi/2) q[37];
cx q[36],q[37];
rz(-pi/2) q[37];
cx q[36],q[37];
rz(-pi/2) q[40];
sx q[40];
rz(-pi/4) q[40];
sx q[40];
rz(-pi/2) q[40];
rz(-pi/2) q[41];
sx q[41];
rz(-pi/4) q[41];
sx q[41];
rz(-pi/2) q[41];
rz(-pi/2) q[42];
sx q[42];
rz(-pi/4) q[42];
sx q[42];
rz(-pi/2) q[42];
cx q[37],q[42];
rz(-pi/2) q[42];
cx q[37],q[42];
cx q[42],q[41];
rz(-pi/2) q[41];
cx q[42],q[41];
cx q[41],q[40];
rz(-pi/2) q[40];
cx q[41],q[40];
rz(-pi/2) q[43];
sx q[43];
rz(-pi/4) q[43];
sx q[43];
rz(-pi/2) q[43];
cx q[40],q[43];
rz(-pi/2) q[43];
cx q[40],q[43];
rz(-pi/2) q[44];
sx q[44];
rz(-pi/4) q[44];
sx q[44];
rz(-pi/2) q[44];
cx q[43],q[44];
rz(-pi/2) q[44];
cx q[43],q[44];
rz(-pi/2) q[45];
sx q[45];
rz(-pi/4) q[45];
sx q[45];
rz(-pi/2) q[45];
cx q[44],q[45];
rz(-pi/2) q[45];
cx q[44],q[45];
rz(-pi/2) q[46];
sx q[46];
rz(-pi/4) q[46];
sx q[46];
rz(-pi/2) q[46];
cx q[45],q[46];
rz(-pi/2) q[46];
cx q[45],q[46];
rz(-pi/2) q[47];
sx q[47];
rz(-pi/4) q[47];
sx q[47];
rz(-pi/2) q[47];
cx q[46],q[47];
rz(-pi/2) q[47];
cx q[46],q[47];
rz(-pi/2) q[50];
sx q[50];
rz(-pi/4) q[50];
sx q[50];
rz(-pi/2) q[50];
rz(-pi/2) q[51];
sx q[51];
rz(-pi/4) q[51];
sx q[51];
rz(-pi/2) q[51];
rz(-pi/2) q[52];
sx q[52];
rz(-pi/4) q[52];
sx q[52];
rz(-pi/2) q[52];
cx q[47],q[52];
rz(-pi/2) q[52];
cx q[47],q[52];
cx q[52],q[51];
rz(-pi/2) q[51];
cx q[52],q[51];
cx q[51],q[50];
rz(-pi/2) q[50];
cx q[51],q[50];
rz(-pi/2) q[53];
sx q[53];
rz(-pi/4) q[53];
sx q[53];
rz(-pi/2) q[53];
cx q[50],q[53];
rz(-pi/2) q[53];
cx q[50],q[53];
rz(-pi/2) q[54];
sx q[54];
rz(-pi/4) q[54];
sx q[54];
rz(-pi/2) q[54];
cx q[53],q[54];
rz(-pi/2) q[54];
cx q[53],q[54];
rz(-pi/2) q[55];
sx q[55];
rz(-pi/4) q[55];
sx q[55];
rz(-pi/2) q[55];
cx q[54],q[55];
rz(-pi/2) q[55];
cx q[54],q[55];
rz(-pi/2) q[56];
sx q[56];
rz(-pi/4) q[56];
sx q[56];
rz(-pi/2) q[56];
cx q[55],q[56];
rz(-pi/2) q[56];
cx q[55],q[56];
rz(-pi/2) q[57];
sx q[57];
rz(-pi/4) q[57];
sx q[57];
rz(-pi/2) q[57];
cx q[56],q[57];
rz(-pi/2) q[57];
cx q[56],q[57];
rz(-pi/2) q[60];
sx q[60];
rz(-pi/4) q[60];
sx q[60];
rz(-pi/2) q[60];
rz(-pi/2) q[61];
sx q[61];
rz(-pi/4) q[61];
sx q[61];
rz(-pi/2) q[61];
rz(-pi/2) q[62];
sx q[62];
rz(-pi/4) q[62];
sx q[62];
rz(-pi/2) q[62];
cx q[57],q[62];
rz(-pi/2) q[62];
cx q[57],q[62];
cx q[62],q[61];
rz(-pi/2) q[61];
cx q[62],q[61];
cx q[61],q[60];
rz(-pi/2) q[60];
cx q[61],q[60];
rz(-pi/2) q[63];
sx q[63];
rz(-pi/4) q[63];
sx q[63];
rz(-pi/2) q[63];
cx q[60],q[63];
rz(-pi/2) q[63];
cx q[60],q[63];
rz(-pi/2) q[64];
sx q[64];
rz(-pi/4) q[64];
sx q[64];
rz(-pi/2) q[64];
cx q[63],q[64];
rz(-pi/2) q[64];
cx q[63],q[64];
rz(-pi/2) q[65];
sx q[65];
rz(-pi/4) q[65];
sx q[65];
rz(-pi/2) q[65];
cx q[64],q[65];
rz(-pi/2) q[65];
cx q[64],q[65];
rz(-pi/2) q[66];
sx q[66];
rz(-pi/4) q[66];
sx q[66];
rz(-pi/2) q[66];
cx q[65],q[66];
rz(-pi/2) q[66];
cx q[65],q[66];
rz(-pi/2) q[67];
sx q[67];
rz(-pi/4) q[67];
sx q[67];
rz(-pi/2) q[67];
cx q[66],q[67];
rz(-pi/2) q[67];
cx q[66],q[67];
rz(-pi/2) q[70];
sx q[70];
rz(-pi/4) q[70];
sx q[70];
rz(-pi/2) q[70];
rz(-pi/2) q[71];
sx q[71];
rz(-pi/4) q[71];
sx q[71];
rz(-pi/2) q[71];
rz(-pi/2) q[72];
sx q[72];
rz(-pi/4) q[72];
sx q[72];
rz(-pi/2) q[72];
cx q[67],q[72];
rz(-pi/2) q[72];
cx q[67],q[72];
cx q[72],q[71];
rz(-pi/2) q[71];
cx q[72],q[71];
cx q[71],q[70];
rz(-pi/2) q[70];
cx q[71],q[70];
rz(-pi/2) q[73];
sx q[73];
rz(-pi/4) q[73];
sx q[73];
rz(-pi/2) q[73];
cx q[70],q[73];
rz(-pi/2) q[73];
cx q[70],q[73];
rz(-pi/2) q[74];
sx q[74];
rz(-pi/4) q[74];
sx q[74];
rz(-pi/2) q[74];
cx q[73],q[74];
rz(-pi/2) q[74];
cx q[73],q[74];
rz(-pi/2) q[75];
sx q[75];
rz(-pi/4) q[75];
sx q[75];
rz(-pi/2) q[75];
cx q[74],q[75];
rz(-pi/2) q[75];
cx q[74],q[75];
rz(-pi/2) q[76];
sx q[76];
rz(-pi/4) q[76];
sx q[76];
rz(-pi/2) q[76];
cx q[75],q[76];
rz(-pi/2) q[76];
cx q[75],q[76];
rz(-pi/2) q[77];
sx q[77];
rz(-pi/4) q[77];
sx q[77];
rz(-pi/2) q[77];
cx q[76],q[77];
rz(-pi/2) q[77];
cx q[76],q[77];
rz(-pi/2) q[80];
sx q[80];
rz(-pi/4) q[80];
sx q[80];
rz(-pi/2) q[80];
rz(-pi/2) q[81];
sx q[81];
rz(-pi/4) q[81];
sx q[81];
rz(-pi/2) q[81];
rz(-pi/2) q[82];
sx q[82];
rz(-pi/4) q[82];
sx q[82];
rz(-pi/2) q[82];
cx q[77],q[82];
rz(-pi/2) q[82];
cx q[77],q[82];
cx q[82],q[81];
rz(-pi/2) q[81];
cx q[82],q[81];
cx q[81],q[80];
rz(-pi/2) q[80];
cx q[81],q[80];
rz(-pi/2) q[83];
sx q[83];
rz(-pi/4) q[83];
sx q[83];
rz(-pi/2) q[83];
cx q[80],q[83];
rz(-pi/2) q[83];
cx q[80],q[83];
rz(-pi/2) q[84];
sx q[84];
rz(-pi/4) q[84];
sx q[84];
rz(-pi/2) q[84];
cx q[83],q[84];
rz(-pi/2) q[84];
cx q[83],q[84];
rz(-pi/2) q[85];
sx q[85];
rz(-pi/4) q[85];
sx q[85];
rz(-pi/2) q[85];
cx q[84],q[85];
rz(-pi/2) q[85];
cx q[84],q[85];
rz(-pi/2) q[86];
sx q[86];
rz(-pi/4) q[86];
sx q[86];
rz(-pi/2) q[86];
cx q[85],q[86];
rz(-pi/2) q[86];
cx q[85],q[86];
rz(-pi/2) q[87];
sx q[87];
rz(-pi/4) q[87];
sx q[87];
rz(-pi/2) q[87];
cx q[86],q[87];
rz(-pi/2) q[87];
cx q[86],q[87];
rz(-pi/2) q[90];
sx q[90];
rz(-pi/4) q[90];
sx q[90];
rz(-pi/2) q[90];
rz(-pi/2) q[91];
sx q[91];
rz(-pi/4) q[91];
sx q[91];
rz(-pi/2) q[91];
rz(-pi/2) q[92];
sx q[92];
rz(-pi/4) q[92];
sx q[92];
rz(-pi/2) q[92];
cx q[87],q[92];
rz(-pi/2) q[92];
cx q[87],q[92];
cx q[92],q[91];
rz(-pi/2) q[91];
cx q[92],q[91];
cx q[91],q[90];
rz(-pi/2) q[90];
cx q[91],q[90];
rz(-pi/2) q[93];
sx q[93];
rz(-pi/4) q[93];
sx q[93];
rz(-pi/2) q[93];
cx q[90],q[93];
rz(-pi/2) q[93];
cx q[90],q[93];
rz(-pi/2) q[94];
sx q[94];
rz(-pi/4) q[94];
sx q[94];
rz(-pi/2) q[94];
cx q[93],q[94];
rz(-pi/2) q[94];
cx q[93],q[94];
rz(-pi/2) q[95];
sx q[95];
rz(-pi/4) q[95];
sx q[95];
rz(-pi/2) q[95];
cx q[94],q[95];
rz(-pi/2) q[95];
cx q[94],q[95];
rz(-pi/2) q[96];
sx q[96];
rz(-pi/4) q[96];
sx q[96];
rz(-pi/2) q[96];
cx q[95],q[96];
rz(-pi/2) q[96];
cx q[95],q[96];
rz(-pi/2) q[97];
sx q[97];
rz(-pi/4) q[97];
sx q[97];
rz(-pi/2) q[97];
cx q[96],q[97];
rz(-pi/2) q[97];
cx q[96],q[97];
rz(-pi/2) q[100];
sx q[100];
rz(-pi/4) q[100];
sx q[100];
rz(-pi/2) q[100];
rz(-pi/2) q[101];
sx q[101];
rz(-pi/4) q[101];
sx q[101];
rz(-pi/2) q[101];
rz(-pi/2) q[102];
sx q[102];
rz(-pi/4) q[102];
sx q[102];
rz(-pi/2) q[102];
cx q[97],q[102];
rz(-pi/2) q[102];
cx q[97],q[102];
cx q[102],q[101];
rz(-pi/2) q[101];
cx q[102],q[101];
cx q[101],q[100];
rz(-pi/2) q[100];
cx q[101],q[100];
rz(-pi/2) q[103];
sx q[103];
rz(-pi/4) q[103];
sx q[103];
rz(-pi/2) q[103];
cx q[100],q[103];
rz(-pi/2) q[103];
cx q[100],q[103];
rz(-pi/2) q[104];
sx q[104];
rz(-pi/4) q[104];
sx q[104];
rz(-pi/2) q[104];
cx q[103],q[104];
rz(-pi/2) q[104];
cx q[103],q[104];
rz(-pi/2) q[105];
sx q[105];
rz(-pi/4) q[105];
sx q[105];
rz(-pi/2) q[105];
cx q[104],q[105];
rz(-pi/2) q[105];
cx q[104],q[105];
rz(-pi/2) q[106];
sx q[106];
rz(-pi/4) q[106];
sx q[106];
rz(-pi/2) q[106];
cx q[105],q[106];
rz(-pi/2) q[106];
cx q[105],q[106];
rz(-pi/2) q[107];
sx q[107];
rz(-pi/4) q[107];
sx q[107];
rz(-pi/2) q[107];
cx q[106],q[107];
rz(-pi/2) q[107];
cx q[106],q[107];
rz(-pi/2) q[110];
sx q[110];
rz(-pi/4) q[110];
sx q[110];
rz(-pi/2) q[110];
rz(-pi/2) q[111];
sx q[111];
rz(-pi/4) q[111];
sx q[111];
rz(-pi/2) q[111];
rz(-pi/2) q[112];
sx q[112];
rz(-pi/4) q[112];
sx q[112];
rz(-pi/2) q[112];
cx q[107],q[112];
rz(-pi/2) q[112];
cx q[107],q[112];
cx q[112],q[111];
rz(-pi/2) q[111];
cx q[112],q[111];
cx q[111],q[110];
rz(-pi/2) q[110];
cx q[111],q[110];
rz(-pi/2) q[113];
sx q[113];
rz(-pi/4) q[113];
sx q[113];
rz(-pi/2) q[113];
cx q[110],q[113];
rz(-pi/2) q[113];
cx q[110],q[113];
rz(-pi/2) q[114];
sx q[114];
rz(-pi/4) q[114];
sx q[114];
rz(-pi/2) q[114];
cx q[113],q[114];
rz(-pi/2) q[114];
cx q[113],q[114];
rz(-pi/2) q[115];
sx q[115];
rz(-pi/4) q[115];
sx q[115];
rz(-pi/2) q[115];
cx q[114],q[115];
rz(-pi/2) q[115];
cx q[114],q[115];
rz(-pi/2) q[116];
sx q[116];
rz(-pi/4) q[116];
sx q[116];
rz(-pi/2) q[116];
cx q[115],q[116];
rz(-pi/2) q[116];
cx q[115],q[116];
rz(-pi/2) q[117];
sx q[117];
rz(-pi/4) q[117];
sx q[117];
rz(-pi/2) q[117];
cx q[116],q[117];
rz(-pi/2) q[117];
cx q[116],q[117];
rz(-pi/2) q[120];
sx q[120];
rz(-pi/4) q[120];
sx q[120];
rz(-pi/2) q[120];
rz(-pi/2) q[121];
sx q[121];
rz(-pi/4) q[121];
sx q[121];
rz(-pi/2) q[121];
rz(-pi/2) q[122];
sx q[122];
rz(-pi/4) q[122];
sx q[122];
rz(-pi/2) q[122];
cx q[117],q[122];
rz(-pi/2) q[122];
cx q[117],q[122];
cx q[122],q[121];
rz(-pi/2) q[121];
cx q[122],q[121];
cx q[121],q[120];
rz(-pi/2) q[120];
cx q[121],q[120];
rz(-pi/2) q[123];
sx q[123];
rz(-pi/4) q[123];
sx q[123];
rz(-pi/2) q[123];
cx q[120],q[123];
rz(-pi/2) q[123];
cx q[120],q[123];
rz(-pi/2) q[124];
sx q[124];
rz(-pi/4) q[124];
sx q[124];
rz(-pi/2) q[124];
cx q[123],q[124];
rz(-pi/2) q[124];
cx q[123],q[124];
rz(-pi/2) q[125];
sx q[125];
rz(-pi/4) q[125];
sx q[125];
rz(-pi/2) q[125];
cx q[124],q[125];
rz(-pi/2) q[125];
cx q[124],q[125];
rz(-pi/2) q[126];
sx q[126];
rz(-pi/4) q[126];
sx q[126];
rz(-pi/2) q[126];
cx q[125],q[126];
rz(-pi/2) q[126];
cx q[125],q[126];
rz(-pi/2) q[127];
sx q[127];
rz(-pi/4) q[127];
sx q[127];
rz(-pi/2) q[127];
cx q[126],q[127];
rz(-pi/2) q[127];
cx q[126],q[127];
rz(-pi/2) q[130];
sx q[130];
rz(-pi/4) q[130];
sx q[130];
rz(-pi/2) q[130];
rz(-pi/2) q[131];
sx q[131];
rz(-pi/4) q[131];
sx q[131];
rz(-pi/2) q[131];
rz(-pi/2) q[132];
sx q[132];
rz(-pi/4) q[132];
sx q[132];
rz(-pi/2) q[132];
cx q[127],q[132];
rz(-pi/2) q[132];
cx q[127],q[132];
cx q[132],q[131];
rz(-pi/2) q[131];
cx q[132],q[131];
cx q[131],q[130];
rz(-pi/2) q[130];
cx q[131],q[130];
rz(-pi/2) q[133];
sx q[133];
rz(-pi/4) q[133];
sx q[133];
rz(-pi/2) q[133];
cx q[130],q[133];
rz(-pi/2) q[133];
cx q[130],q[133];
rz(-pi/2) q[134];
sx q[134];
rz(-pi/4) q[134];
sx q[134];
rz(-pi/2) q[134];
cx q[133],q[134];
rz(-pi/2) q[134];
cx q[133],q[134];
rz(-pi/2) q[135];
sx q[135];
rz(-pi/4) q[135];
sx q[135];
rz(-pi/2) q[135];
cx q[134],q[135];
rz(-pi/2) q[135];
cx q[134],q[135];
rz(-pi/2) q[136];
sx q[136];
rz(-pi/4) q[136];
sx q[136];
rz(-pi/2) q[136];
cx q[135],q[136];
rz(-pi/2) q[136];
cx q[135],q[136];
rz(-pi/2) q[137];
sx q[137];
rz(-pi/4) q[137];
sx q[137];
rz(-pi/2) q[137];
cx q[136],q[137];
rz(-pi/2) q[137];
cx q[136],q[137];
rz(-pi/2) q[140];
sx q[140];
rz(-pi/4) q[140];
sx q[140];
rz(-pi/2) q[140];
rz(-pi/2) q[141];
sx q[141];
rz(-pi/4) q[141];
sx q[141];
rz(-pi/2) q[141];
rz(-pi/2) q[142];
sx q[142];
rz(-pi/4) q[142];
sx q[142];
rz(-pi/2) q[142];
cx q[137],q[142];
rz(-pi/2) q[142];
cx q[137],q[142];
cx q[142],q[141];
rz(-pi/2) q[141];
cx q[142],q[141];
cx q[141],q[140];
rz(-pi/2) q[140];
cx q[141],q[140];
rz(-pi/2) q[143];
sx q[143];
rz(-pi/4) q[143];
sx q[143];
rz(-pi/2) q[143];
cx q[140],q[143];
rz(-pi/2) q[143];
cx q[140],q[143];
rz(-pi/2) q[144];
sx q[144];
rz(-pi/4) q[144];
sx q[144];
rz(-pi/2) q[144];
cx q[143],q[144];
rz(-pi/2) q[144];
cx q[143],q[144];
rz(-pi/2) q[145];
sx q[145];
rz(-pi/4) q[145];
sx q[145];
rz(-pi/2) q[145];
cx q[144],q[145];
rz(-pi/2) q[145];
cx q[144],q[145];
rz(-pi/2) q[146];
sx q[146];
rz(-pi/4) q[146];
sx q[146];
rz(-pi/2) q[146];
cx q[145],q[146];
rz(-pi/2) q[146];
cx q[145],q[146];
rz(-pi/2) q[147];
sx q[147];
rz(-pi/4) q[147];
sx q[147];
rz(-pi/2) q[147];
cx q[146],q[147];
rz(-pi/2) q[147];
cx q[146],q[147];
rz(-pi/2) q[150];
sx q[150];
rz(-pi/4) q[150];
sx q[150];
rz(-pi/2) q[150];
rz(-pi/2) q[151];
sx q[151];
rz(-pi/4) q[151];
sx q[151];
rz(-pi/2) q[151];
rz(-pi/2) q[152];
sx q[152];
rz(-pi/4) q[152];
sx q[152];
rz(-pi/2) q[152];
cx q[147],q[152];
rz(-pi/2) q[152];
cx q[147],q[152];
cx q[152],q[151];
rz(-pi/2) q[151];
cx q[152],q[151];
cx q[151],q[150];
rz(-pi/2) q[150];
cx q[151],q[150];
rz(-pi/2) q[153];
sx q[153];
rz(-pi/4) q[153];
sx q[153];
rz(-pi/2) q[153];
cx q[150],q[153];
rz(-pi/2) q[153];
cx q[150],q[153];
rz(-pi/2) q[154];
sx q[154];
rz(-pi/4) q[154];
sx q[154];
rz(-pi/2) q[154];
cx q[153],q[154];
rz(-pi/2) q[154];
cx q[153],q[154];
rz(-pi/2) q[155];
sx q[155];
rz(-pi/4) q[155];
sx q[155];
rz(-pi/2) q[155];
cx q[154],q[155];
rz(-pi/2) q[155];
cx q[154],q[155];
rz(-pi/2) q[156];
sx q[156];
rz(-pi/4) q[156];
sx q[156];
rz(-pi/2) q[156];
cx q[155],q[156];
rz(-pi/2) q[156];
cx q[155],q[156];
rz(-pi/2) q[157];
sx q[157];
rz(-pi/4) q[157];
sx q[157];
rz(-pi/2) q[157];
cx q[156],q[157];
rz(-pi/2) q[157];
cx q[156],q[157];
rz(-pi/2) q[160];
sx q[160];
rz(-pi/4) q[160];
sx q[160];
rz(-pi/2) q[160];
rz(-pi/2) q[161];
sx q[161];
rz(-pi/4) q[161];
sx q[161];
rz(-pi/2) q[161];
rz(-pi/2) q[162];
sx q[162];
rz(-pi/4) q[162];
sx q[162];
rz(-pi/2) q[162];
cx q[157],q[162];
rz(-pi/2) q[162];
cx q[157],q[162];
cx q[162],q[161];
rz(-pi/2) q[161];
cx q[162],q[161];
cx q[161],q[160];
rz(-pi/2) q[160];
cx q[161],q[160];
rz(-pi/2) q[163];
sx q[163];
rz(-pi/4) q[163];
sx q[163];
rz(-pi/2) q[163];
cx q[160],q[163];
rz(-pi/2) q[163];
cx q[160],q[163];
rz(-pi/2) q[164];
sx q[164];
rz(-pi/4) q[164];
sx q[164];
rz(-pi/2) q[164];
cx q[163],q[164];
rz(-pi/2) q[164];
cx q[163],q[164];
rz(-pi/2) q[165];
sx q[165];
rz(-pi/4) q[165];
sx q[165];
rz(-pi/2) q[165];
cx q[164],q[165];
rz(-pi/2) q[165];
cx q[164],q[165];
rz(-pi/2) q[166];
sx q[166];
rz(-pi/4) q[166];
sx q[166];
rz(-pi/2) q[166];
cx q[165],q[166];
rz(-pi/2) q[166];
cx q[165],q[166];
rz(-pi/2) q[167];
sx q[167];
rz(-pi/4) q[167];
sx q[167];
rz(-pi/2) q[167];
cx q[166],q[167];
rz(-pi/2) q[167];
cx q[166],q[167];
measure q[2] -> m0[0];
measure q[1] -> m0[1];
measure q[0] -> m0[2];
measure q[3] -> m0[3];
measure q[4] -> m0[4];
measure q[5] -> m0[5];
measure q[6] -> m0[6];
measure q[7] -> m0[7];
measure q[12] -> m0[8];
measure q[11] -> m0[9];
measure q[10] -> m0[10];
measure q[13] -> m0[11];
measure q[14] -> m0[12];
measure q[15] -> m0[13];
measure q[16] -> m0[14];
measure q[17] -> m0[15];
measure q[22] -> m0[16];
measure q[21] -> m0[17];
measure q[20] -> m0[18];
measure q[23] -> m0[19];
measure q[24] -> m0[20];
measure q[25] -> m0[21];
measure q[26] -> m0[22];
measure q[27] -> m0[23];
measure q[32] -> m0[24];
measure q[31] -> m0[25];
measure q[30] -> m0[26];
measure q[33] -> m0[27];
measure q[34] -> m0[28];
measure q[35] -> m0[29];
measure q[36] -> m0[30];
measure q[37] -> m0[31];
measure q[42] -> m0[32];
measure q[41] -> m0[33];
measure q[40] -> m0[34];
measure q[43] -> m0[35];
measure q[44] -> m0[36];
measure q[45] -> m0[37];
measure q[46] -> m0[38];
measure q[47] -> m0[39];
measure q[52] -> m0[40];
measure q[51] -> m0[41];
measure q[50] -> m0[42];
measure q[53] -> m0[43];
measure q[54] -> m0[44];
measure q[55] -> m0[45];
measure q[56] -> m0[46];
measure q[57] -> m0[47];
measure q[62] -> m0[48];
measure q[61] -> m0[49];
measure q[60] -> m0[50];
measure q[63] -> m0[51];
measure q[64] -> m0[52];
measure q[65] -> m0[53];
measure q[66] -> m0[54];
measure q[67] -> m0[55];
measure q[72] -> m0[56];
measure q[71] -> m0[57];
measure q[70] -> m0[58];
measure q[73] -> m0[59];
measure q[74] -> m0[60];
measure q[75] -> m0[61];
measure q[76] -> m0[62];
measure q[77] -> m0[63];
measure q[82] -> m0[64];
measure q[81] -> m0[65];
measure q[80] -> m0[66];
measure q[83] -> m0[67];
measure q[84] -> m0[68];
measure q[85] -> m0[69];
measure q[86] -> m0[70];
measure q[87] -> m0[71];
measure q[92] -> m0[72];
measure q[91] -> m0[73];
measure q[90] -> m0[74];
measure q[93] -> m0[75];
measure q[94] -> m0[76];
measure q[95] -> m0[77];
measure q[96] -> m0[78];
measure q[97] -> m0[79];
measure q[102] -> m0[80];
measure q[101] -> m0[81];
measure q[100] -> m0[82];
measure q[103] -> m0[83];
measure q[104] -> m0[84];
measure q[105] -> m0[85];
measure q[106] -> m0[86];
measure q[107] -> m0[87];
measure q[112] -> m0[88];
measure q[111] -> m0[89];
measure q[110] -> m0[90];
measure q[113] -> m0[91];
measure q[114] -> m0[92];
measure q[115] -> m0[93];
measure q[116] -> m0[94];
measure q[117] -> m0[95];
measure q[122] -> m0[96];
measure q[121] -> m0[97];
measure q[120] -> m0[98];
measure q[123] -> m0[99];
measure q[124] -> m0[100];
measure q[125] -> m0[101];
measure q[126] -> m0[102];
measure q[127] -> m0[103];
measure q[132] -> m0[104];
measure q[131] -> m0[105];
measure q[130] -> m0[106];
measure q[133] -> m0[107];
measure q[134] -> m0[108];
measure q[135] -> m0[109];
measure q[136] -> m0[110];
measure q[137] -> m0[111];
measure q[142] -> m0[112];
measure q[141] -> m0[113];
measure q[140] -> m0[114];
measure q[143] -> m0[115];
measure q[144] -> m0[116];
measure q[145] -> m0[117];
measure q[146] -> m0[118];
measure q[147] -> m0[119];
measure q[152] -> m0[120];
measure q[151] -> m0[121];
measure q[150] -> m0[122];
measure q[153] -> m0[123];
measure q[154] -> m0[124];
measure q[155] -> m0[125];
measure q[156] -> m0[126];
measure q[157] -> m0[127];
measure q[162] -> m0[128];
measure q[161] -> m0[129];
measure q[160] -> m0[130];
measure q[163] -> m0[131];
measure q[164] -> m0[132];
measure q[165] -> m0[133];
measure q[166] -> m0[134];
measure q[167] -> m0[135];
