OPENQASM 2.0;
include "qelib1.inc";
qreg q[100];
rz(pi/2) q[97];
sx q[97];
rz(pi/2) q[97];
cx q[97],q[96];
cx q[96],q[95];
cx q[95],q[94];
cx q[94],q[89];
cx q[89],q[86];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[83];
cx q[83],q[80];
cx q[80],q[81];
cx q[81],q[82];
cx q[82],q[88];
cx q[88],q[90];
cx q[90],q[91];
cx q[91],q[92];
cx q[92],q[77];
cx q[77],q[76];
cx q[76],q[75];
cx q[75],q[74];
cx q[74],q[69];
cx q[69],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[60];
cx q[60],q[61];
cx q[61],q[62];
cx q[62],q[68];
cx q[68],q[70];
cx q[70],q[71];
cx q[71],q[72];
cx q[72],q[57];
cx q[57],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[49];
cx q[49],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[40];
cx q[40],q[41];
cx q[41],q[42];
cx q[42],q[48];
cx q[48],q[50];
cx q[50],q[51];
cx q[51],q[52];
cx q[52],q[37];
cx q[37],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[29];
cx q[29],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[20];
cx q[20],q[21];
cx q[21],q[22];
cx q[22],q[28];
cx q[28],q[30];
cx q[30],q[31];
cx q[31],q[32];
cx q[32],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[14];
cx q[14],q[9];
cx q[9],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[0];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[8];
cx q[8],q[10];
cx q[10],q[11];
cx q[11],q[12];
