OPENQASM 2.0;
include "qelib1.inc";
qreg q[80];
rz(-pi/2) q[12];
sx q[12];
rz(-pi/2) q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[8];
rz(-pi) q[10];
x q[10];
rz(-pi) q[11];
x q[11];
x q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[8],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[0],q[3];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
cx q[3],q[4];
rz(-pi) q[3];
x q[3];
cx q[4],q[5];
rz(-pi) q[4];
x q[4];
cx q[5],q[6];
rz(-pi) q[5];
x q[5];
cx q[6],q[9];
rz(-pi) q[6];
x q[6];
rz(-pi) q[8];
x q[8];
cx q[10],q[8];
cx q[8],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[0],q[3];
cx q[3],q[4];
cx q[4],q[5];
cx q[5],q[6];
cx q[9],q[14];
rz(-pi) q[9];
x q[9];
cx q[6],q[9];
cx q[14],q[15];
rz(-pi) q[14];
x q[14];
cx q[9],q[14];
cx q[15],q[16];
rz(-pi) q[15];
x q[15];
cx q[14],q[15];
cx q[16],q[17];
rz(-pi) q[16];
x q[16];
cx q[15],q[16];
cx q[17],q[32];
rz(-pi) q[17];
x q[17];
cx q[16],q[17];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[28];
cx q[28],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[23];
rz(-pi) q[20];
x q[20];
rz(-pi) q[21];
x q[21];
rz(-pi) q[22];
x q[22];
cx q[23],q[24];
rz(-pi) q[23];
x q[23];
cx q[24],q[25];
rz(-pi) q[24];
x q[24];
cx q[25],q[26];
rz(-pi) q[25];
x q[25];
cx q[26],q[29];
rz(-pi) q[26];
x q[26];
rz(-pi) q[28];
x q[28];
rz(-pi) q[30];
x q[30];
rz(-pi) q[31];
x q[31];
rz(-pi) q[32];
x q[32];
cx q[17],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[28];
cx q[28],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[23];
cx q[23],q[24];
cx q[24],q[25];
cx q[25],q[26];
cx q[29],q[34];
rz(-pi) q[29];
x q[29];
cx q[26],q[29];
cx q[34],q[35];
rz(-pi) q[34];
x q[34];
cx q[29],q[34];
cx q[35],q[36];
rz(-pi) q[35];
x q[35];
cx q[34],q[35];
cx q[36],q[37];
rz(-pi) q[36];
x q[36];
cx q[35],q[36];
cx q[37],q[52];
rz(-pi) q[37];
x q[37];
cx q[36],q[37];
cx q[52],q[51];
cx q[37],q[52];
cx q[51],q[50];
cx q[50],q[48];
cx q[48],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[43];
cx q[43],q[44];
cx q[44],q[45];
cx q[45],q[46];
cx q[46],q[49];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[48];
cx q[48],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[43];
cx q[43],q[44];
cx q[44],q[45];
cx q[45],q[46];
cx q[49],q[54];
cx q[46],q[49];
cx q[54],q[55];
cx q[49],q[54];
cx q[55],q[56];
cx q[54],q[55];
cx q[56],q[57];
cx q[55],q[56];
cx q[57],q[72];
cx q[56],q[57];
cx q[72],q[71];
cx q[57],q[72];
cx q[71],q[70];
cx q[70],q[68];
cx q[68],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[63];
cx q[63],q[64];
cx q[64],q[65];
cx q[65],q[66];
cx q[66],q[69];
cx q[72],q[71];
cx q[71],q[70];
cx q[70],q[68];
cx q[68],q[62];
cx q[62],q[61];
cx q[61],q[60];
cx q[60],q[63];
cx q[63],q[64];
cx q[64],q[65];
cx q[65],q[66];
cx q[69],q[74];
cx q[66],q[69];
cx q[74],q[75];
cx q[69],q[74];
cx q[75],q[76];
cx q[74],q[75];
cx q[76],q[77];
cx q[75],q[76];
cx q[76],q[77];
