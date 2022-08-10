OPENQASM 2.0;
include "qelib1.inc";
qreg q[60];
rz(pi/2) q[3];
sx q[3];
rz(-pi/2) q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
cx q[22],q[7];
cx q[7],q[22];
cx q[22],q[7];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[20];
cx q[20],q[23];
cx q[23],q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
rz(pi/2) q[28];
sx q[28];
rz(pi/2) q[28];
rz(pi/2) q[29];
sx q[29];
rz(pi/2) q[29];
rz(pi/2) q[30];
sx q[30];
rz(pi/2) q[30];
rz(pi/2) q[31];
sx q[31];
rz(pi/2) q[31];
rz(pi/2) q[32];
sx q[32];
rz(pi/2) q[32];
rz(pi/2) q[33];
sx q[33];
rz(pi/2) q[33];
rz(pi/2) q[34];
sx q[34];
rz(pi/2) q[34];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
rz(pi/2) q[36];
sx q[36];
rz(pi/2) q[36];
rz(pi/2) q[37];
sx q[37];
rz(pi/2) q[37];
rz(pi/2) q[38];
sx q[38];
rz(pi/2) q[38];
cx q[32],q[38];
cx q[38],q[32];
cx q[32],q[38];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[32],q[38];
cx q[38],q[32];
cx q[32],q[38];
rz(pi/2) q[39];
sx q[39];
rz(pi/2) q[39];
cx q[36],q[39];
cx q[39],q[36];
cx q[36],q[39];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
rz(pi/2) q[41];
sx q[41];
rz(pi/2) q[41];
rz(pi/2) q[42];
sx q[42];
rz(pi/2) q[42];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[40];
cx q[40],q[43];
cx q[43],q[40];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[27],q[26];
cx q[26],q[27];
cx q[27],q[26];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[34],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
rz(pi/2) q[48];
sx q[48];
rz(pi/2) q[48];
rz(pi/2) q[49];
sx q[49];
rz(pi/2) q[49];
rz(pi/2) q[50];
sx q[50];
rz(pi/2) q[50];
rz(pi/2) q[51];
sx q[51];
rz(pi/2) q[51];
rz(pi/2) q[52];
sx q[52];
rz(pi/2) q[52];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[50],q[48];
cx q[48],q[50];
cx q[50],q[48];
rz(pi/2) q[55];
sx q[55];
rz(pi/2) q[55];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[50],q[53];
cx q[53],q[50];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[49],q[54];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[49],q[54];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[55],q[54];
cx q[54],q[55];
cx q[55],q[54];
cx q[54],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[45],q[46];
cx q[46],q[49];
cx q[49],q[46];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
cx q[54],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[46],q[45];
cx q[45],q[46];
cx q[44],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[46],q[49];
cx q[49],q[46];
cx q[46],q[49];
cx q[54],q[49];
cx q[49],q[54];
cx q[54],q[49];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[54];
cx q[54],q[53];
cx q[53],q[50];
cx q[50],q[53];
cx q[53],q[50];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
cx q[52],q[58];
cx q[58],q[52];
cx q[52],q[58];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[48],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[42],q[48];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[48],q[50];
cx q[50],q[48];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[52],q[58];
cx q[58],q[52];
cx q[52],q[58];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[50],q[48];
cx q[48],q[50];
cx q[42],q[48];
cx q[42],q[27];
cx q[27],q[42];
cx q[42],q[27];
cx q[48],q[50];
cx q[50],q[48];
cx q[42],q[48];
cx q[48],q[42];
cx q[42],q[48];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[36],q[39];
cx q[39],q[36];
cx q[36],q[39];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[50],q[51];
cx q[51],q[50];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[37],q[52];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[51],q[50];
cx q[50],q[51];
cx q[48],q[50];
cx q[50],q[51];
cx q[51],q[50];
cx q[52],q[51];
cx q[51],q[52];
cx q[52],q[51];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[52];
cx q[52],q[37];
cx q[37],q[36];
cx q[36],q[37];
cx q[37],q[36];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[36];
cx q[36],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[29],q[34];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[30];
cx q[30],q[33];
cx q[31],q[30];
cx q[30],q[33];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[32],q[17];
cx q[17],q[32];
cx q[32],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[19];
cx q[19],q[16];
cx q[16],q[19];
cx q[33],q[30];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[33],q[34];
cx q[34],q[33];
cx q[29],q[34];
cx q[34],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[25],q[26];
cx q[24],q[25];
cx q[25],q[24];
cx q[26],q[29];
cx q[29],q[26];
cx q[26],q[29];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[34],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[34],q[33];
cx q[33],q[34];
cx q[30],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[34],q[29];
cx q[29],q[34];
cx q[26],q[29];
cx q[29],q[34];
cx q[34],q[29];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[30];
cx q[30],q[33];
cx q[33],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[17],q[32];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[16],q[15];
cx q[15],q[16];
cx q[16],q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[14],q[9];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[28];
cx q[28],q[30];
cx q[22],q[28];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[28],q[30];
cx q[30],q[28];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[32],q[38];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[38],q[32];
cx q[32],q[38];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[33],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[17],q[32];
cx q[17],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[28];
cx q[28],q[30];
cx q[22],q[28];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[28],q[30];
cx q[30],q[28];
cx q[22],q[28];
cx q[28],q[22];
cx q[22],q[28];
cx q[32],q[38];
cx q[38],q[32];
cx q[32],q[38];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[30],q[31];
cx q[31],q[30];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[17],q[32];
cx q[32],q[17];
cx q[17],q[32];
cx q[32],q[17];
cx q[17],q[16];
cx q[16],q[17];
cx q[15],q[16];
cx q[16],q[17];
cx q[17],q[16];
cx q[32],q[17];
cx q[17],q[32];
cx q[32],q[17];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[28],q[30];
cx q[30],q[28];
cx q[28],q[30];
cx q[30],q[28];
cx q[28],q[22];
cx q[22],q[28];
cx q[22],q[7];
cx q[7],q[22];
cx q[9],q[14];
cx q[14],q[9];
cx q[6],q[9];
cx q[9],q[6];
cx q[6],q[9];
cx q[14],q[9];
cx q[6],q[7];
cx q[7],q[6];
cx q[6],q[7];
cx q[7],q[6];
cx q[9],q[14];
cx q[14],q[9];
cx q[9],q[6];
cx q[14],q[9];
cx q[9],q[14];
cx q[14],q[9];
cx q[9],q[6];
cx q[5],q[6];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[4],q[5];
barrier q[23],q[40],q[58],q[43],q[55],q[46],q[53],q[45],q[49],q[44],q[54],q[51],q[42],q[41],q[39],q[27],q[50],q[37],q[48],q[52],q[36],q[25],q[19],q[35],q[29],q[24],q[34],q[26],q[32],q[13],q[20],q[38],q[33],q[16],q[21],q[31],q[17],q[15],q[30],q[28],q[7],q[22],q[14],q[9],q[3],q[6],q[4],q[5];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
rz(pi/2) q[14];
sx q[14];
rz(pi/2) q[14];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
rz(pi/2) q[17];
sx q[17];
rz(pi/2) q[17];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
rz(pi/2) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/2) q[22];
sx q[22];
rz(pi/2) q[22];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
rz(pi/2) q[24];
sx q[24];
rz(pi/2) q[24];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/2) q[26];
sx q[26];
rz(pi/2) q[26];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
rz(pi/2) q[28];
sx q[28];
rz(pi/2) q[28];
rz(pi/2) q[29];
sx q[29];
rz(pi/2) q[29];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/2) q[30];
sx q[30];
rz(pi/2) q[30];
rz(pi/2) q[31];
sx q[31];
rz(pi/2) q[31];
rz(pi/2) q[32];
sx q[32];
rz(pi/2) q[32];
rz(pi/2) q[33];
sx q[33];
rz(pi/2) q[33];
rz(pi/2) q[34];
sx q[34];
rz(pi/2) q[34];
rz(pi/2) q[35];
sx q[35];
rz(pi/2) q[35];
rz(pi/2) q[36];
sx q[36];
rz(pi/2) q[36];
rz(pi/2) q[37];
sx q[37];
rz(pi/2) q[37];
rz(pi/2) q[38];
sx q[38];
rz(pi/2) q[38];
rz(pi/2) q[39];
sx q[39];
rz(pi/2) q[39];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/2) q[40];
sx q[40];
rz(pi/2) q[40];
rz(pi/2) q[41];
sx q[41];
rz(pi/2) q[41];
rz(pi/2) q[42];
sx q[42];
rz(pi/2) q[42];
rz(pi/2) q[43];
sx q[43];
rz(pi/2) q[43];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
rz(pi/2) q[46];
sx q[46];
rz(pi/2) q[46];
rz(pi/2) q[48];
sx q[48];
rz(pi/2) q[48];
rz(pi/2) q[49];
sx q[49];
rz(pi/2) q[49];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/2) q[50];
sx q[50];
rz(pi/2) q[50];
rz(pi/2) q[51];
sx q[51];
rz(pi/2) q[51];
rz(pi/2) q[52];
sx q[52];
rz(pi/2) q[52];
rz(pi/2) q[53];
sx q[53];
rz(pi/2) q[53];
rz(pi/2) q[54];
sx q[54];
rz(pi/2) q[54];
rz(pi/2) q[55];
sx q[55];
rz(pi/2) q[55];
rz(pi/2) q[58];
sx q[58];
rz(pi/2) q[58];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
