OPENQASM 2.0;
include "qelib1.inc";
qreg q[90];
rz(pi/2) q[3];
sx q[3];
rz(-3*pi/4) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
rz(pi/2) q[6];
sx q[6];
rz(-3*pi/4) q[6];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[4],q[5];
rz(-pi/2) q[4];
sx q[4];
rz(-3*pi/4) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
rz(-pi) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[13];
sx q[13];
rz(-3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(-3*pi/4) q[14];
cx q[14],q[9];
cx q[9],q[14];
cx q[14],q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[15],q[14];
x q[14];
sx q[15];
rz(pi/2) q[16];
sx q[16];
rz(-3*pi/4) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[17];
sx q[17];
rz(-3*pi/4) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[19];
sx q[19];
rz(-3*pi/4) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
rz(3*pi/4) q[16];
sx q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/2) q[24];
sx q[24];
rz(-3*pi/4) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[24],q[19];
cx q[19],q[24];
cx q[24],q[19];
rz(pi/2) q[24];
rz(pi/2) q[30];
sx q[30];
rz(pi/2) q[30];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[32],q[7];
cx q[7],q[32];
cx q[32],q[7];
cx q[6],q[7];
sx q[6];
x q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[9];
rz(3*pi/4) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[9],q[6];
cx q[6],q[9];
cx q[14],q[9];
cx q[9],q[14];
cx q[14],q[9];
cx q[14],q[15];
sx q[14];
rz(-pi) q[15];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
sx q[19];
rz(pi/2) q[19];
cx q[9],q[6];
rz(-pi) q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
sx q[5];
rz(pi/2) q[5];
rz(-pi/2) q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[6],q[9];
cx q[14],q[9];
sx q[6];
rz(-3*pi/4) q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
sx q[4];
rz(pi/2) q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[9],q[14];
cx q[14],q[9];
cx q[14],q[15];
sx q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
x q[15];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
rz(3*pi/4) q[9];
sx q[9];
cx q[14],q[9];
cx q[9],q[14];
cx q[14],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[6],q[9];
cx q[14],q[9];
cx q[5],q[6];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[9],q[14];
cx q[14],q[9];
rz(pi/2) q[9];
cx q[33],q[30];
cx q[30],q[33];
cx q[33],q[30];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
rz(pi/2) q[36];
sx q[36];
rz(pi/2) q[36];
rz(pi/2) q[38];
sx q[38];
rz(pi/2) q[38];
cx q[32],q[38];
cx q[38],q[32];
cx q[32],q[38];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[33];
rz(-pi/2) q[30];
sx q[30];
rz(-3*pi/4) q[30];
rz(-pi) q[33];
sx q[33];
rz(pi/2) q[33];
cx q[33],q[30];
cx q[30],q[33];
cx q[33],q[30];
sx q[33];
rz(pi/2) q[33];
rz(pi/2) q[39];
sx q[39];
rz(pi/2) q[39];
cx q[36],q[39];
cx q[39],q[36];
cx q[36],q[39];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[34];
rz(-pi) q[34];
sx q[34];
rz(pi/2) q[34];
sx q[35];
rz(3*pi/4) q[35];
sx q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
rz(pi/2) q[34];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
rz(pi/2) q[40];
sx q[40];
rz(-3*pi/4) q[40];
rz(pi/2) q[41];
sx q[41];
rz(-3*pi/4) q[41];
rz(pi/2) q[42];
sx q[42];
rz(-3*pi/4) q[42];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
cx q[40],q[38];
cx q[38],q[40];
cx q[40],q[38];
cx q[32],q[38];
cx q[38],q[32];
cx q[32],q[38];
cx q[32],q[31];
rz(-pi) q[31];
sx q[31];
rz(pi/2) q[31];
rz(-pi/2) q[32];
sx q[32];
rz(-3*pi/4) q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
sx q[30];
rz(pi/2) q[30];
sx q[43];
rz(pi/2) q[43];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[44],q[39];
cx q[39],q[44];
cx q[44],q[39];
cx q[36],q[39];
cx q[39],q[36];
cx q[36],q[39];
cx q[36],q[37];
rz(-pi/2) q[36];
sx q[36];
rz(-3*pi/4) q[36];
rz(-pi) q[37];
sx q[37];
rz(pi/2) q[37];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
sx q[37];
rz(pi/2) q[37];
rz(pi/2) q[45];
sx q[45];
rz(-3*pi/4) q[45];
rz(pi/2) q[46];
sx q[46];
rz(-3*pi/4) q[46];
rz(pi/2) q[47];
sx q[47];
rz(pi/2) q[47];
cx q[47],q[46];
cx q[46],q[47];
cx q[47],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[39];
x q[39];
sx q[44];
cx q[44],q[39];
cx q[39],q[44];
cx q[44],q[39];
rz(3*pi/4) q[39];
sx q[39];
cx q[36],q[39];
cx q[39],q[36];
cx q[36],q[39];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
rz(pi/2) q[35];
sx q[47];
rz(pi/2) q[47];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[38],q[40];
sx q[38];
rz(3*pi/4) q[38];
sx q[38];
cx q[32],q[38];
cx q[38],q[32];
cx q[32],q[38];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
rz(pi/2) q[31];
cx q[32],q[38];
cx q[38],q[32];
cx q[32],q[38];
rz(-pi) q[40];
sx q[40];
rz(pi/2) q[40];
sx q[41];
rz(pi/2) q[41];
sx q[42];
rz(pi/2) q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
sx q[48];
rz(pi/2) q[48];
rz(pi/2) q[49];
sx q[49];
rz(-3*pi/4) q[49];
rz(pi/2) q[50];
sx q[50];
rz(-3*pi/4) q[50];
sx q[50];
rz(pi/2) q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[40],q[38];
cx q[38],q[40];
cx q[40],q[38];
cx q[32],q[38];
sx q[32];
rz(3*pi/4) q[32];
sx q[32];
rz(pi/2) q[32];
x q[38];
rz(pi/2) q[51];
sx q[51];
rz(-3*pi/4) q[51];
sx q[51];
rz(pi/2) q[51];
rz(pi/2) q[52];
sx q[52];
rz(-3*pi/4) q[52];
sx q[52];
rz(pi/2) q[52];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[42],q[17];
x q[17];
sx q[42];
cx q[42],q[17];
cx q[17],q[42];
cx q[42],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(3*pi/4) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[48],q[50];
sx q[48];
x q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[17];
cx q[17],q[42];
cx q[42],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[17],q[16];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
rz(pi/2) q[42];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[50],q[51];
rz(pi/2) q[50];
rz(3*pi/4) q[52];
sx q[52];
rz(pi/2) q[52];
rz(pi/2) q[53];
sx q[53];
rz(-3*pi/4) q[53];
sx q[53];
rz(pi/2) q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[54];
sx q[54];
rz(-3*pi/4) q[54];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[45],q[46];
sx q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(3*pi/4) q[44];
sx q[44];
cx q[44],q[39];
cx q[39],q[44];
cx q[44],q[39];
cx q[36],q[39];
cx q[39],q[36];
cx q[36],q[39];
rz(pi/2) q[36];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
cx q[40],q[41];
rz(-pi/2) q[40];
cx q[40],q[38];
cx q[38],q[40];
cx q[40],q[38];
sx q[38];
rz(-3*pi/4) q[38];
sx q[38];
rz(pi/2) q[38];
rz(-pi) q[41];
sx q[41];
rz(pi/2) q[41];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
sx q[41];
rz(pi/2) q[41];
cx q[42],q[48];
cx q[44],q[39];
cx q[39],q[44];
cx q[44],q[39];
cx q[44],q[43];
rz(-pi) q[43];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
sx q[40];
rz(pi/2) q[40];
sx q[44];
cx q[44],q[39];
cx q[39],q[44];
cx q[44],q[39];
rz(3*pi/4) q[39];
sx q[39];
rz(pi/2) q[39];
cx q[43],q[44];
rz(pi/2) q[43];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
sx q[40];
rz(pi/2) q[40];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
x q[46];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[50];
rz(pi/2) q[48];
sx q[48];
rz(pi/2) q[48];
sx q[49];
rz(pi/2) q[49];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[46],q[45];
cx q[45],q[46];
sx q[54];
rz(pi/2) q[54];
sx q[55];
rz(pi/2) q[55];
rz(pi/2) q[56];
sx q[56];
rz(-3*pi/4) q[56];
sx q[56];
rz(pi/2) q[56];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[46],q[49];
rz(-pi/2) q[46];
cx q[45],q[46];
cx q[46],q[45];
sx q[45];
rz(-3*pi/4) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
x q[49];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[46],q[47];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[54],q[55];
rz(-pi/2) q[54];
x q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[49];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
sx q[55];
rz(-3*pi/4) q[55];
sx q[55];
rz(pi/2) q[55];
rz(pi/2) q[57];
sx q[57];
rz(-3*pi/4) q[57];
sx q[57];
rz(pi/2) q[57];
rz(pi/2) q[59];
sx q[59];
rz(-3*pi/4) q[59];
sx q[59];
rz(pi/2) q[59];
cx q[56],q[59];
cx q[59],q[56];
cx q[56],q[59];
rz(pi/2) q[60];
sx q[60];
rz(pi/2) q[60];
rz(pi/2) q[61];
sx q[61];
rz(-3*pi/4) q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
cx q[63],q[60];
cx q[60],q[63];
cx q[63],q[60];
sx q[63];
rz(pi/2) q[63];
rz(pi/2) q[64];
sx q[64];
rz(-3*pi/4) q[64];
sx q[64];
rz(pi/2) q[64];
rz(pi/2) q[65];
sx q[65];
rz(-3*pi/4) q[65];
sx q[65];
rz(pi/2) q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
rz(pi/2) q[66];
sx q[66];
rz(-3*pi/4) q[66];
sx q[66];
rz(pi/2) q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
rz(pi/2) q[67];
sx q[67];
rz(-3*pi/4) q[67];
sx q[67];
rz(pi/2) q[67];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
rz(pi/2) q[69];
sx q[69];
rz(-3*pi/4) q[69];
sx q[69];
rz(pi/2) q[69];
rz(pi/2) q[70];
sx q[70];
rz(-3*pi/4) q[70];
rz(pi/2) q[71];
sx q[71];
rz(pi/2) q[71];
rz(pi/2) q[72];
sx q[72];
rz(pi/2) q[72];
rz(pi/2) q[73];
sx q[73];
rz(pi/2) q[73];
cx q[73],q[70];
cx q[70],q[73];
cx q[73],q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
cx q[61],q[60];
x q[60];
sx q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(3*pi/4) q[60];
sx q[60];
cx q[63],q[60];
cx q[60],q[63];
cx q[63],q[60];
rz(pi/2) q[63];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
cx q[68],q[62];
rz(-pi) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
sx q[61];
rz(pi/2) q[61];
cx q[61],q[60];
rz(-pi) q[60];
sx q[60];
rz(pi/2) q[60];
sx q[61];
rz(3*pi/4) q[61];
sx q[61];
cx q[61],q[60];
cx q[60],q[61];
cx q[61],q[60];
rz(pi/2) q[60];
rz(-pi/2) q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
sx q[62];
rz(-3*pi/4) q[62];
sx q[62];
rz(pi/2) q[62];
cx q[62],q[61];
cx q[61],q[62];
cx q[62],q[61];
sx q[73];
rz(pi/2) q[73];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
rz(pi/2) q[77];
sx q[77];
rz(-3*pi/4) q[77];
sx q[77];
rz(pi/2) q[77];
rz(pi/2) q[78];
sx q[78];
rz(-3*pi/4) q[78];
rz(pi/2) q[79];
sx q[79];
rz(pi/2) q[79];
cx q[76],q[79];
cx q[79],q[76];
cx q[76],q[79];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[74];
x q[74];
sx q[75];
rz(pi/2) q[80];
sx q[80];
rz(-3*pi/4) q[80];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[78];
cx q[78],q[80];
cx q[80],q[78];
cx q[72],q[78];
cx q[78],q[72];
cx q[72],q[78];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[70],q[71];
rz(-pi/2) q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
sx q[68];
rz(-3*pi/4) q[68];
sx q[68];
rz(pi/2) q[68];
cx q[62],q[68];
cx q[68],q[62];
cx q[62],q[68];
rz(-pi) q[71];
cx q[78],q[72];
x q[72];
rz(-pi/2) q[78];
cx q[72],q[78];
cx q[78],q[72];
cx q[72],q[78];
sx q[80];
rz(pi/2) q[80];
sx q[81];
rz(pi/2) q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[78];
cx q[78],q[80];
cx q[80],q[78];
cx q[72],q[78];
cx q[78],q[72];
cx q[72],q[78];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[70],q[71];
rz(-pi/2) q[70];
x q[71];
cx q[71],q[70];
cx q[70],q[71];
cx q[71],q[70];
cx q[70],q[73];
rz(pi/2) q[70];
sx q[70];
rz(pi/2) q[70];
cx q[70],q[68];
cx q[68],q[70];
cx q[70],q[68];
sx q[71];
rz(-3*pi/4) q[71];
sx q[71];
rz(pi/2) q[71];
sx q[72];
rz(pi/2) q[72];
sx q[78];
rz(-3*pi/4) q[78];
cx q[72],q[78];
cx q[78],q[72];
cx q[72],q[78];
sx q[72];
rz(pi/2) q[72];
rz(pi/2) q[82];
sx q[82];
rz(-3*pi/4) q[82];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[57];
cx q[57],q[82];
cx q[82],q[57];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[80],q[81];
rz(-pi/2) q[80];
cx q[80],q[78];
cx q[78],q[80];
cx q[80],q[78];
sx q[78];
rz(-3*pi/4) q[78];
sx q[78];
rz(pi/2) q[78];
rz(-pi) q[81];
rz(pi/2) q[83];
sx q[83];
rz(-3*pi/4) q[83];
sx q[83];
rz(pi/2) q[83];
rz(pi/2) q[84];
sx q[84];
rz(pi/2) q[84];
cx q[84],q[79];
cx q[79],q[84];
cx q[84],q[79];
cx q[79],q[76];
x q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[69],q[66];
rz(-pi) q[66];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
sx q[64];
rz(pi/2) q[64];
rz(-pi/2) q[69];
cx q[75],q[74];
x q[74];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[66],q[65];
cx q[65],q[64];
cx q[64],q[65];
cx q[65],q[64];
rz(pi/2) q[66];
cx q[67],q[66];
cx q[66],q[67];
cx q[67],q[66];
sx q[67];
rz(pi/2) q[67];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
sx q[69];
rz(-3*pi/4) q[69];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
cx q[66],q[65];
cx q[65],q[66];
cx q[66],q[65];
sx q[65];
rz(pi/2) q[65];
rz(-pi/2) q[75];
cx q[77],q[76];
cx q[76],q[77];
cx q[77],q[76];
rz(3*pi/4) q[77];
sx q[77];
rz(pi/2) q[77];
rz(-pi/2) q[79];
cx q[76],q[79];
cx q[79],q[76];
cx q[76],q[79];
sx q[76];
rz(-3*pi/4) q[76];
rz(pi/2) q[85];
sx q[85];
rz(-3*pi/4) q[85];
rz(pi/2) q[86];
sx q[86];
rz(pi/2) q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[84];
rz(-pi) q[84];
rz(-pi/2) q[85];
rz(pi/2) q[88];
sx q[88];
rz(-3*pi/4) q[88];
sx q[88];
rz(pi/2) q[88];
cx q[82],q[88];
cx q[88],q[82];
cx q[82],q[88];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[80],q[81];
rz(-pi/2) q[80];
sx q[80];
rz(-3*pi/4) q[80];
x q[81];
sx q[82];
rz(pi/2) q[82];
cx q[82],q[57];
cx q[57],q[82];
cx q[82],q[57];
cx q[57],q[56];
cx q[56],q[59];
rz(pi/2) q[57];
sx q[57];
rz(pi/2) q[57];
cx q[59],q[56];
cx q[56],q[59];
cx q[56],q[55];
cx q[55],q[56];
cx q[56],q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[46],q[49];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
cx q[82],q[81];
cx q[81],q[82];
cx q[82],q[81];
cx q[81],q[80];
cx q[80],q[81];
cx q[81],q[80];
cx q[80],q[78];
cx q[78],q[80];
cx q[80],q[78];
cx q[72],q[78];
cx q[78],q[72];
cx q[72],q[78];
cx q[72],q[71];
cx q[71],q[72];
cx q[72],q[71];
cx q[70],q[71];
rz(pi/2) q[70];
sx q[70];
rz(pi/2) q[70];
sx q[81];
rz(pi/2) q[81];
cx q[82],q[88];
rz(pi/2) q[82];
sx q[82];
rz(pi/2) q[82];
rz(pi/2) q[89];
sx q[89];
rz(pi/2) q[89];
cx q[86],q[89];
cx q[89],q[86];
cx q[86],q[89];
cx q[86],q[87];
rz(-pi/2) q[86];
x q[87];
cx q[87],q[86];
cx q[86],q[87];
cx q[87],q[86];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[79];
rz(-pi) q[79];
rz(-pi/2) q[84];
sx q[85];
rz(pi/2) q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[79];
cx q[79],q[84];
cx q[84],q[79];
cx q[76],q[79];
cx q[79],q[76];
cx q[76],q[79];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[74];
rz(-pi) q[74];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
sx q[66];
rz(pi/2) q[66];
rz(-pi/2) q[75];
sx q[76];
rz(-3*pi/4) q[76];
sx q[79];
rz(pi/2) q[79];
sx q[84];
rz(pi/2) q[84];
cx q[84],q[79];
cx q[79],q[84];
cx q[84],q[79];
cx q[76],q[79];
cx q[79],q[76];
cx q[76],q[79];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[84],q[83];
cx q[83],q[84];
cx q[84],q[83];
cx q[84],q[79];
cx q[79],q[84];
cx q[84],q[79];
cx q[76],q[79];
cx q[79],q[76];
cx q[76],q[79];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
cx q[69],q[74];
rz(pi/2) q[69];
cx q[66],q[69];
cx q[69],q[66];
cx q[66],q[69];
sx q[66];
rz(pi/2) q[66];
cx q[74],q[69];
cx q[69],q[74];
cx q[74],q[69];
cx q[76],q[75];
cx q[75],q[74];
cx q[74],q[75];
cx q[75],q[74];
rz(pi/2) q[76];
cx q[76],q[75];
cx q[75],q[76];
cx q[76],q[75];
sx q[75];
rz(pi/2) q[75];
sx q[79];
rz(-3*pi/4) q[79];
sx q[84];
rz(pi/2) q[84];
sx q[85];
rz(-3*pi/4) q[85];
sx q[85];
rz(pi/2) q[85];
sx q[86];
rz(-3*pi/4) q[86];
sx q[86];
rz(pi/2) q[86];
sx q[87];
rz(-3*pi/4) q[87];
sx q[87];
rz(pi/2) q[87];
sx q[89];
rz(pi/2) q[89];
cx q[86],q[89];
cx q[89],q[86];
cx q[86],q[89];
cx q[86],q[85];
cx q[85],q[86];
cx q[86],q[85];
cx q[85],q[84];
cx q[84],q[85];
cx q[85],q[84];
cx q[84],q[79];
cx q[79],q[84];
cx q[84],q[79];
cx q[76],q[79];
rz(pi/2) q[76];
sx q[76];
rz(pi/2) q[76];
sx q[84];
rz(pi/2) q[84];
