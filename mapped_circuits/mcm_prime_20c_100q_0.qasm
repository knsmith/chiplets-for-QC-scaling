OPENQASM 2.0;
include "qelib1.inc";
qreg q[100];
rz(pi/2) q[0];
sx q[0];
rz(-3*pi/4) q[0];
rz(pi/2) q[2];
sx q[2];
rz(-3*pi/4) q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
rz(pi/2) q[4];
sx q[4];
rz(-3*pi/4) q[4];
rz(pi/2) q[5];
sx q[5];
rz(-3*pi/4) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-3*pi/4) q[1];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[2];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
rz(-pi/2) q[3];
sx q[3];
rz(-3*pi/4) q[3];
sx q[4];
rz(pi/2) q[4];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[1],q[2];
sx q[1];
rz(3*pi/4) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
rz(pi/2) q[2];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[9];
sx q[9];
rz(-3*pi/4) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[10];
sx q[10];
rz(-3*pi/4) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/2) q[12];
sx q[12];
rz(-3*pi/4) q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/2) q[13];
sx q[13];
rz(-3*pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[0],q[7];
sx q[0];
rz(3*pi/4) q[0];
sx q[0];
rz(-pi) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[1],q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[7];
cx q[9],q[10];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
rz(-pi) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
sx q[12];
rz(pi/2) q[12];
sx q[14];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[17];
sx q[17];
rz(-3*pi/4) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[18];
sx q[18];
cx q[18],q[6];
sx q[18];
rz(-3*pi/4) q[18];
sx q[18];
rz(pi/2) q[18];
rz(-pi) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[4],q[8];
rz(-pi/2) q[4];
sx q[5];
rz(pi/2) q[5];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
x q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[5];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
sx q[8];
rz(-3*pi/4) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[15],q[19];
sx q[15];
x q[19];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
rz(-pi) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
sx q[12];
rz(pi) q[12];
cx q[12],q[11];
sx q[12];
rz(pi/2) q[12];
sx q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
rz(3*pi/4) q[15];
sx q[15];
cx q[17],q[15];
cx q[15],q[17];
cx q[14],q[15];
sx q[14];
rz(3*pi/4) q[14];
sx q[14];
rz(pi/2) q[14];
x q[15];
rz(pi/2) q[17];
cx q[15],q[17];
cx q[17],q[15];
rz(3*pi/4) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[20];
sx q[20];
rz(-3*pi/4) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[21];
sx q[21];
rz(-3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
rz(pi/2) q[22];
sx q[22];
rz(-3*pi/4) q[22];
cx q[22],q[16];
cx q[16],q[22];
cx q[22],q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[24];
sx q[24];
rz(-3*pi/4) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[16];
cx q[16],q[22];
cx q[22],q[16];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
sx q[23];
x q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[22];
rz(-pi) q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
sx q[21];
rz(pi) q[21];
cx q[21],q[20];
sx q[21];
rz(pi/2) q[21];
sx q[23];
rz(3*pi/4) q[24];
sx q[24];
cx q[26],q[25];
rz(-pi) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[22];
x q[22];
cx q[22],q[16];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(-pi/2) q[23];
sx q[23];
rz(-3*pi/4) q[23];
sx q[23];
rz(pi/2) q[23];
rz(3*pi/4) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[25];
sx q[26];
rz(3*pi/4) q[26];
sx q[26];
rz(pi/2) q[26];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
rz(pi/2) q[29];
sx q[29];
rz(-3*pi/4) q[29];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
sx q[27];
rz(pi/2) q[27];
rz(pi/2) q[30];
sx q[30];
rz(pi/2) q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
rz(pi/2) q[31];
sx q[31];
rz(pi/2) q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[29],q[30];
sx q[29];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
rz(3*pi/4) q[27];
sx q[27];
rz(pi/2) q[27];
rz(-pi) q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
sx q[29];
rz(pi/2) q[29];
rz(pi/2) q[33];
sx q[33];
rz(-3*pi/4) q[33];
cx q[33],q[28];
cx q[28],q[33];
cx q[33],q[28];
sx q[28];
rz(pi/2) q[28];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[31],q[32];
rz(-pi/2) q[31];
x q[32];
rz(pi/2) q[34];
sx q[34];
rz(-3*pi/4) q[34];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[33],q[34];
rz(-pi/2) q[33];
x q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
rz(-pi/2) q[33];
sx q[34];
rz(-3*pi/4) q[34];
sx q[35];
rz(pi/2) q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
rz(pi/2) q[36];
sx q[36];
rz(-3*pi/4) q[36];
sx q[36];
rz(pi/2) q[36];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
rz(-pi) q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
sx q[29];
rz(pi/2) q[29];
sx q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
rz(3*pi/4) q[30];
sx q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
rz(pi/2) q[29];
x q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
sx q[31];
rz(-3*pi/4) q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
sx q[30];
rz(pi/2) q[30];
cx q[34],q[33];
cx q[33],q[34];
cx q[32],q[33];
sx q[32];
x q[33];
sx q[36];
rz(-3*pi/4) q[36];
sx q[36];
rz(pi/2) q[36];
rz(pi/2) q[37];
sx q[37];
rz(-3*pi/4) q[37];
sx q[37];
rz(pi/2) q[37];
cx q[37],q[35];
cx q[35],q[37];
cx q[37],q[35];
cx q[34],q[35];
rz(pi/2) q[34];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[28];
cx q[28],q[33];
cx q[33],q[28];
sx q[28];
rz(pi/2) q[28];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[31],q[32];
rz(pi/2) q[31];
sx q[31];
rz(pi/2) q[31];
rz(3*pi/4) q[33];
sx q[33];
rz(pi/2) q[33];
sx q[37];
rz(pi/2) q[37];
rz(pi/2) q[38];
sx q[38];
rz(-3*pi/4) q[38];
sx q[38];
rz(pi/2) q[38];
cx q[26],q[38];
cx q[38],q[26];
cx q[26],q[38];
cx q[17],q[26];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[39];
sx q[39];
rz(-3*pi/4) q[39];
sx q[39];
rz(pi/2) q[39];
cx q[35],q[39];
cx q[39],q[35];
cx q[35],q[39];
cx q[34],q[35];
rz(pi/2) q[34];
sx q[34];
rz(pi/2) q[34];
rz(pi/2) q[40];
sx q[40];
rz(-3*pi/4) q[40];
rz(pi/2) q[41];
sx q[41];
rz(-3*pi/4) q[41];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
rz(pi/2) q[44];
sx q[44];
rz(-3*pi/4) q[44];
rz(pi/2) q[46];
sx q[46];
rz(-3*pi/4) q[46];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
cx q[47],q[40];
cx q[40],q[47];
cx q[47],q[40];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
sx q[40];
rz(pi/2) q[40];
sx q[47];
rz(pi/2) q[47];
cx q[48],q[44];
cx q[44],q[48];
cx q[48],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[41],q[42];
sx q[41];
rz(3*pi/4) q[41];
sx q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
rz(-pi) q[42];
sx q[42];
rz(pi/2) q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[43],q[44];
sx q[43];
rz(3*pi/4) q[43];
sx q[43];
rz(-pi) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[47],q[40];
cx q[40],q[47];
cx q[47],q[40];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[42],q[41];
x q[41];
sx q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
rz(3*pi/4) q[41];
sx q[41];
rz(pi/2) q[41];
sx q[48];
rz(pi/2) q[48];
rz(pi/2) q[49];
sx q[49];
rz(-3*pi/4) q[49];
sx q[49];
rz(pi/2) q[49];
cx q[49],q[47];
cx q[47],q[49];
cx q[40],q[47];
sx q[40];
rz(3*pi/4) q[40];
sx q[40];
rz(pi/2) q[40];
x q[47];
rz(pi/2) q[49];
cx q[47],q[49];
cx q[49],q[47];
rz(pi/2) q[50];
sx q[50];
rz(-3*pi/4) q[50];
sx q[50];
rz(pi/2) q[50];
rz(pi/2) q[51];
sx q[51];
rz(-3*pi/4) q[51];
rz(pi/2) q[52];
sx q[52];
rz(-3*pi/4) q[52];
rz(pi/2) q[53];
sx q[53];
rz(-3*pi/4) q[53];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
rz(pi/2) q[55];
sx q[55];
rz(pi/2) q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[52];
cx q[52],q[53];
cx q[53],q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
sx q[52];
rz(pi/2) q[52];
sx q[53];
rz(pi/2) q[53];
sx q[54];
rz(pi/2) q[54];
rz(pi/2) q[56];
sx q[56];
rz(-3*pi/4) q[56];
rz(pi/2) q[57];
sx q[57];
rz(pi/2) q[57];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[46],q[58];
cx q[58],q[46];
cx q[46],q[58];
cx q[46],q[45];
x q[45];
sx q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
rz(3*pi/4) q[45];
sx q[45];
rz(-pi/2) q[46];
sx q[58];
rz(pi/2) q[58];
x q[58];
cx q[58],q[46];
cx q[46],q[58];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
rz(pi/2) q[44];
rz(pi/2) q[45];
rz(pi/2) q[46];
cx q[48],q[44];
cx q[44],q[48];
cx q[48],q[44];
cx q[43],q[44];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
sx q[48];
rz(pi/2) q[48];
sx q[58];
rz(-3*pi/4) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[55],q[59];
sx q[55];
x q[59];
cx q[55],q[59];
cx q[59],q[55];
cx q[55],q[59];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
rz(3*pi/4) q[59];
sx q[59];
rz(pi/2) q[59];
rz(pi/2) q[61];
sx q[61];
rz(-3*pi/4) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(pi/2) q[62];
sx q[62];
rz(-3*pi/4) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[62],q[56];
cx q[56],q[62];
cx q[62],q[56];
cx q[51],q[56];
sx q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
rz(3*pi/4) q[50];
sx q[50];
rz(pi/2) q[50];
cx q[50],q[49];
cx q[49],q[50];
cx q[50],q[49];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[53],q[52];
rz(-pi) q[52];
sx q[52];
rz(pi/2) q[52];
rz(-pi/2) q[53];
sx q[53];
rz(-3*pi/4) q[53];
cx q[53],q[52];
cx q[52],q[53];
cx q[53],q[52];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[52];
cx q[52],q[53];
cx q[53],q[52];
sx q[53];
rz(pi/2) q[53];
cx q[54],q[55];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
rz(-pi) q[56];
sx q[56];
rz(pi/2) q[56];
cx q[56],q[51];
x q[51];
sx q[56];
cx q[56],q[51];
cx q[51],q[56];
cx q[56],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
rz(3*pi/4) q[50];
sx q[50];
rz(pi/2) q[50];
cx q[51],q[52];
rz(pi/2) q[51];
sx q[51];
rz(pi/2) q[51];
rz(pi/2) q[63];
sx q[63];
rz(-3*pi/4) q[63];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
rz(pi/2) q[67];
sx q[67];
rz(-3*pi/4) q[67];
rz(pi/2) q[68];
sx q[68];
rz(-3*pi/4) q[68];
cx q[68],q[64];
cx q[64],q[68];
cx q[68],q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
cx q[63],q[64];
cx q[64],q[63];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
sx q[63];
rz(pi/2) q[63];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
sx q[65];
rz(pi/2) q[65];
cx q[66],q[57];
cx q[57],q[66];
cx q[66],q[57];
sx q[57];
rz(pi/2) q[57];
rz(pi/2) q[69];
sx q[69];
rz(pi/2) q[69];
cx q[69],q[67];
cx q[67],q[69];
cx q[69],q[67];
cx q[67],q[60];
cx q[60],q[67];
cx q[67],q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
sx q[60];
rz(pi/2) q[60];
cx q[61],q[62];
sx q[61];
rz(-pi) q[62];
sx q[67];
rz(pi/2) q[67];
cx q[67],q[60];
cx q[60],q[67];
cx q[67],q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(3*pi/4) q[60];
sx q[60];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
sx q[61];
rz(pi/2) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[60],q[67];
sx q[60];
rz(pi/2) q[61];
cx q[63],q[62];
cx q[62],q[63];
cx q[63],q[62];
rz(-pi) q[67];
sx q[69];
rz(pi/2) q[69];
cx q[69],q[67];
cx q[67],q[69];
cx q[69],q[67];
cx q[67],q[60];
cx q[60],q[67];
cx q[67],q[60];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(3*pi/4) q[67];
sx q[67];
rz(pi/2) q[67];
sx q[69];
rz(pi) q[69];
rz(pi/2) q[70];
sx q[70];
rz(-3*pi/4) q[70];
sx q[70];
rz(pi/2) q[70];
cx q[69],q[70];
sx q[69];
rz(pi/2) q[69];
rz(pi/2) q[71];
sx q[71];
rz(-3*pi/4) q[71];
rz(pi/2) q[73];
sx q[73];
rz(-3*pi/4) q[73];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
rz(pi/2) q[74];
sx q[74];
rz(pi/2) q[74];
rz(pi/2) q[75];
sx q[75];
rz(-3*pi/4) q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
sx q[74];
rz(pi/2) q[74];
rz(pi/2) q[76];
sx q[76];
rz(pi/2) q[76];
cx q[76],q[71];
cx q[71],q[76];
cx q[76],q[71];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
sx q[71];
rz(pi/2) q[71];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[73],q[68];
cx q[68],q[73];
cx q[73],q[68];
cx q[68],q[64];
rz(-pi) q[64];
sx q[68];
cx q[73],q[72];
x q[72];
sx q[73];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
rz(3*pi/4) q[72];
sx q[72];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
rz(pi/2) q[71];
cx q[73],q[68];
cx q[68],q[73];
cx q[73],q[68];
cx q[68],q[64];
cx q[64],q[68];
cx q[68],q[64];
sx q[68];
rz(pi/2) q[68];
rz(3*pi/4) q[73];
sx q[73];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
rz(pi/2) q[72];
sx q[76];
rz(pi/2) q[76];
cx q[76],q[71];
cx q[71],q[76];
cx q[76],q[71];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[75],q[77];
rz(-pi/2) q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
sx q[74];
rz(-3*pi/4) q[74];
sx q[74];
rz(pi/2) q[74];
x q[77];
cx q[77],q[75];
rz(-pi) q[75];
sx q[75];
rz(pi) q[75];
rz(-pi/2) q[77];
sx q[77];
rz(-3*pi/4) q[77];
sx q[77];
rz(pi/2) q[77];
cx q[66],q[78];
sx q[66];
x q[78];
cx q[66],q[78];
cx q[78],q[66];
cx q[66],q[78];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[64],q[63];
x q[63];
cx q[63],q[62];
cx q[62],q[56];
cx q[56],q[62];
cx q[62],q[56];
cx q[62],q[61];
rz(pi/2) q[62];
sx q[62];
rz(pi/2) q[62];
rz(pi/2) q[63];
sx q[63];
rz(pi/2) q[63];
sx q[64];
cx q[66],q[57];
cx q[57],q[66];
cx q[66],q[57];
cx q[65],q[66];
rz(-pi/2) q[65];
x q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[66],q[57];
cx q[57],q[66];
cx q[66],q[57];
sx q[57];
rz(-3*pi/4) q[57];
sx q[57];
rz(pi/2) q[57];
cx q[68],q[64];
cx q[64],q[68];
cx q[68],q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[65],q[66];
sx q[65];
rz(3*pi/4) q[65];
sx q[65];
rz(-pi) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
rz(pi/2) q[66];
cx q[73],q[68];
cx q[68],q[73];
cx q[73],q[68];
cx q[64],q[68];
rz(pi/2) q[64];
sx q[64];
rz(pi/2) q[64];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
cx q[68],q[64];
cx q[64],q[68];
cx q[68],q[64];
rz(3*pi/4) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[73],q[72];
cx q[72],q[73];
cx q[73],q[72];
cx q[68],q[73];
rz(pi/2) q[68];
sx q[68];
rz(pi/2) q[68];
rz(3*pi/4) q[78];
sx q[78];
rz(pi/2) q[78];
rz(pi/2) q[79];
sx q[79];
rz(-3*pi/4) q[79];
sx q[79];
rz(pi/2) q[79];
cx q[75],q[79];
sx q[75];
rz(pi/2) q[75];
