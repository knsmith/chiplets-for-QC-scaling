OPENQASM 2.0;
include "qelib1.inc";
qreg q[60];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[5],q[6];
cx q[6],q[5];
cx q[4],q[8];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[9],q[7];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[10],q[9];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[9],q[10];
cx q[10],q[9];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[4],q[8];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[4],q[8];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[8],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[22],q[21];
cx q[21],q[22];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[22],q[23];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[25],q[24];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[28],q[24];
cx q[24],q[28];
cx q[28],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[22],q[23];
cx q[22],q[16];
cx q[16],q[22];
cx q[22],q[16];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[16];
cx q[16],q[22];
cx q[22],q[16];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[22],q[16];
cx q[16],q[22];
cx q[22],q[16];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[30],q[31];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[30],q[31];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[34],q[35];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[26],q[38];
cx q[38],q[26];
cx q[26],q[38];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[24],q[25];
cx q[26],q[38];
cx q[38],q[26];
cx q[26],q[38];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[26],q[38];
cx q[38],q[26];
cx q[26],q[38];
cx q[35],q[39];
cx q[39],q[35];
cx q[35],q[39];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[32],q[33];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[35],q[39];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[39],q[35];
cx q[35],q[39];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[46],q[37];
cx q[37],q[46];
cx q[46],q[37];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
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
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[46],q[37];
cx q[37],q[46];
cx q[46],q[37];
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
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[46],q[58];
cx q[58],q[46];
cx q[46],q[58];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[43],q[44];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[41],q[42];
rz(-pi/4) q[42];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
rz(pi/4) q[42];
cx q[41],q[42];
rz(pi/4) q[41];
rz(-pi/4) q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
rz(-pi/4) q[41];
x q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
rz(pi/4) q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
rz(3*pi/4) q[43];
sx q[43];
rz(pi/2) q[43];
cx q[44],q[43];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[45],q[44];
rz(-pi/4) q[44];
cx q[43],q[44];
rz(pi/4) q[44];
cx q[45],q[44];
rz(-pi/4) q[44];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
rz(3*pi/4) q[43];
sx q[43];
rz(pi/2) q[43];
rz(pi/4) q[45];
cx q[44],q[45];
rz(pi/4) q[44];
rz(-pi/4) q[45];
cx q[44],q[45];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
x q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[44],q[45];
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
cx q[42],q[43];
rz(-pi/4) q[43];
cx q[44],q[43];
rz(pi/4) q[43];
cx q[42],q[43];
rz(pi/4) q[42];
rz(-pi/4) q[43];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
rz(3*pi/4) q[42];
sx q[42];
rz(pi/2) q[42];
cx q[44],q[43];
rz(-pi/4) q[43];
rz(pi/4) q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
x q[42];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[36];
cx q[36],q[42];
cx q[42],q[36];
cx q[46],q[37];
cx q[37],q[46];
cx q[46],q[37];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
rz(pi/2) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[58],q[46];
rz(-pi/4) q[46];
cx q[45],q[46];
rz(pi/4) q[46];
cx q[58],q[46];
rz(-pi/4) q[46];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
rz(3*pi/4) q[45];
sx q[45];
rz(pi/2) q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[42],q[43];
rz(pi/2) q[42];
sx q[42];
rz(pi/2) q[42];
cx q[42],q[36];
cx q[36],q[42];
cx q[42],q[36];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
rz(-pi/4) q[31];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[36];
rz(pi/4) q[36];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
rz(-pi/4) q[31];
rz(pi/4) q[32];
cx q[42],q[36];
cx q[36],q[42];
cx q[42],q[36];
cx q[31],q[36];
cx q[36],q[31];
cx q[31],q[32];
rz(pi/4) q[31];
rz(-pi/4) q[32];
cx q[31],q[32];
rz(3*pi/4) q[36];
sx q[36];
rz(pi/2) q[36];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
x q[31];
cx q[33],q[32];
rz(pi/2) q[33];
sx q[33];
rz(pi/2) q[33];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[35],q[34];
rz(-pi/4) q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[32],q[33];
rz(pi/4) q[33];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[33];
rz(-pi/4) q[33];
cx q[32],q[33];
rz(pi/4) q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[32],q[33];
rz(pi/4) q[32];
rz(-pi/4) q[33];
cx q[32],q[33];
rz(3*pi/4) q[34];
sx q[34];
rz(pi/2) q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
x q[34];
cx q[33],q[34];
cx q[35],q[39];
cx q[36],q[31];
cx q[39],q[35];
cx q[35],q[39];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[35],q[39];
cx q[39],q[35];
cx q[35],q[39];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
cx q[35],q[39];
cx q[39],q[35];
cx q[42],q[36];
cx q[36],q[42];
cx q[42],q[36];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[31],q[32];
rz(pi/2) q[31];
sx q[31];
rz(pi/2) q[31];
cx q[30],q[31];
rz(-pi/4) q[31];
cx q[32],q[31];
rz(pi/4) q[31];
cx q[30],q[31];
rz(pi/4) q[30];
rz(-pi/4) q[31];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
rz(3*pi/4) q[30];
sx q[30];
rz(pi/2) q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[27],q[29];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[32],q[31];
rz(-pi/4) q[31];
x q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
rz(pi/4) q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[32];
rz(-pi/4) q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[30],q[31];
rz(pi/4) q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[33],q[32];
rz(-pi/4) q[32];
cx q[31],q[32];
rz(pi/4) q[33];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[31],q[32];
rz(pi/4) q[31];
rz(-pi/4) q[32];
cx q[31],q[32];
rz(3*pi/4) q[33];
sx q[33];
rz(pi/2) q[33];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[30],q[31];
rz(pi/2) q[30];
sx q[30];
rz(pi/2) q[30];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
x q[33];
cx q[32],q[33];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[32];
rz(-pi/4) q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[30],q[31];
rz(pi/4) q[31];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
rz(-pi/4) q[31];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
rz(3*pi/4) q[30];
sx q[30];
rz(pi/2) q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[21],q[20];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[22],q[21];
rz(-pi/4) q[21];
cx q[20],q[21];
rz(pi/4) q[21];
cx q[22],q[21];
rz(-pi/4) q[21];
cx q[20],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
rz(3*pi/4) q[20];
sx q[20];
rz(pi/2) q[20];
rz(pi/4) q[22];
cx q[21],q[22];
rz(pi/4) q[21];
rz(-pi/4) q[22];
x q[22];
cx q[22],q[16];
cx q[16],q[22];
cx q[22],q[16];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[27],q[20];
rz(pi/2) q[27];
sx q[27];
rz(pi/2) q[27];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[21],q[20];
rz(-pi/4) q[20];
cx q[27],q[20];
rz(pi/4) q[20];
cx q[21],q[20];
rz(-pi/4) q[20];
rz(pi/4) q[21];
cx q[27],q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
rz(3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[25],q[24];
rz(pi/2) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[26],q[25];
rz(-pi/4) q[25];
cx q[24],q[25];
rz(pi/4) q[25];
cx q[26],q[25];
rz(-pi/4) q[25];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
rz(3*pi/4) q[24];
sx q[24];
rz(pi/2) q[24];
cx q[23],q[24];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(pi/4) q[26];
cx q[25],q[26];
rz(pi/4) q[25];
rz(-pi/4) q[26];
x q[26];
cx q[26],q[38];
cx q[27],q[20];
rz(-pi/4) q[20];
x q[20];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
rz(pi/4) q[27];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
rz(pi/4) q[32];
cx q[31],q[32];
rz(pi/4) q[31];
rz(-pi/4) q[32];
x q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[38],q[26];
cx q[26],q[38];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
rz(-pi/4) q[24];
cx q[23],q[24];
rz(pi/4) q[24];
cx q[25],q[24];
rz(-pi/4) q[24];
cx q[23],q[24];
rz(pi/4) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
rz(pi/4) q[23];
rz(-pi/4) q[24];
cx q[23],q[24];
rz(3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
x q[25];
cx q[26],q[38];
cx q[28],q[24];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
rz(pi/2) q[28];
sx q[28];
rz(pi/2) q[28];
cx q[28],q[24];
cx q[24],q[28];
cx q[28],q[24];
cx q[23],q[24];
rz(-pi/4) q[24];
cx q[25],q[24];
rz(pi/4) q[24];
cx q[23],q[24];
rz(pi/4) q[23];
rz(-pi/4) q[24];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(3*pi/4) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[16];
cx q[16],q[22];
cx q[22],q[16];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[10],q[11];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[9];
cx q[25],q[24];
rz(-pi/4) q[24];
x q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
rz(pi/4) q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[28],q[24];
cx q[24],q[28];
cx q[28],q[24];
cx q[38],q[26];
rz(pi/4) q[58];
cx q[46],q[58];
rz(pi/4) q[46];
rz(-pi/4) q[58];
x q[58];
cx q[46],q[58];
cx q[58],q[46];
cx q[46],q[58];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[46],q[58];
cx q[58],q[46];
cx q[46],q[58];
cx q[9],q[10];
cx q[10],q[9];
cx q[7],q[9];
rz(-pi/4) q[9];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[11],q[10];
rz(pi/4) q[10];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[7],q[9];
rz(pi/4) q[7];
rz(-pi/4) q[9];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[10],q[11];
cx q[11],q[10];
rz(3*pi/4) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[11],q[12];
rz(pi/4) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[11],q[12];
rz(pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[11],q[12];
rz(pi/4) q[11];
rz(-pi/4) q[12];
x q[12];
rz(3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[8],q[13];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[14],q[13];
rz(-pi/4) q[13];
cx q[8],q[13];
rz(pi/4) q[13];
cx q[14],q[13];
rz(-pi/4) q[13];
rz(pi/4) q[14];
cx q[8],q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(3*pi/4) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[19],q[15];
rz(pi/2) q[19];
sx q[19];
rz(pi/2) q[19];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[14],q[15];
rz(-pi/4) q[15];
cx q[19],q[15];
rz(pi/4) q[15];
cx q[14],q[15];
rz(pi/4) q[14];
rz(-pi/4) q[15];
cx q[19],q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
rz(3*pi/4) q[14];
sx q[14];
rz(pi/2) q[14];
cx q[19],q[15];
rz(-pi/4) q[15];
x q[15];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
rz(pi/4) q[19];
cx q[8],q[13];
rz(-pi/4) q[13];
x q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
rz(pi/4) q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[10],q[9];
rz(pi/4) q[10];
rz(-pi/4) q[9];
x q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[10],q[9];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[9],q[10];
cx q[10],q[9];
cx q[10],q[11];
rz(pi/2) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[7],q[9];
rz(-pi/4) q[9];
cx q[10],q[9];
rz(pi/4) q[9];
cx q[7],q[9];
rz(pi/4) q[7];
rz(-pi/4) q[9];
cx q[9],q[10];
cx q[10],q[9];
rz(3*pi/4) q[10];
sx q[10];
rz(pi/2) q[10];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[8],q[13];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[3],q[4];
rz(-pi/4) q[4];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
rz(pi/4) q[8];
cx q[4],q[8];
rz(pi/4) q[4];
rz(-pi/4) q[8];
cx q[8],q[13];
cx q[13],q[8];
rz(3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[8],q[4];
rz(-pi/4) q[4];
rz(pi/4) q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[4],q[3];
cx q[3],q[4];
x q[3];
cx q[8],q[13];
cx q[4],q[8];
cx q[3],q[4];
cx q[4],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[8],q[13];
cx q[13],q[8];
cx q[8],q[4];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[6],q[18];
cx q[18],q[6];
cx q[6],q[18];
cx q[8],q[13];
cx q[13],q[8];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[8],q[13];
rz(pi/4) q[13];
cx q[12],q[13];
rz(pi/4) q[12];
rz(-pi/4) q[13];
cx q[8],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[8],q[13];
rz(-pi/4) q[13];
x q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
rz(pi/4) q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[9],q[7];
rz(-pi/4) q[7];
x q[7];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
rz(pi/4) q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[9],q[10];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[9];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[9],q[10];
cx q[10],q[9];
cx q[10],q[11];
rz(pi/4) q[11];
cx q[12],q[11];
rz(-pi/4) q[11];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
rz(3*pi/4) q[10];
sx q[10];
rz(pi/2) q[10];
rz(pi/4) q[12];
cx q[11],q[12];
rz(pi/4) q[11];
rz(-pi/4) q[12];
cx q[11],q[12];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
x q[11];
cx q[10],q[11];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[8],q[13];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[5],q[4];
rz(-pi/4) q[4];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
rz(pi/4) q[8];
cx q[4],q[8];
rz(pi/4) q[4];
rz(-pi/4) q[8];
cx q[13],q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
rz(pi/4) q[13];
rz(3*pi/4) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[5],q[6];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[6],q[18];
cx q[18],q[6];
cx q[6],q[18];
rz(-pi/4) q[8];
x q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[4],q[5];
rz(-pi/4) q[5];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[8],q[13];
cx q[13],q[8];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[8],q[4];
rz(pi/4) q[4];
cx q[5],q[4];
rz(-pi/4) q[4];
cx q[4],q[8];
rz(pi/4) q[5];
cx q[8],q[4];
cx q[4],q[5];
rz(pi/4) q[4];
rz(-pi/4) q[5];
cx q[4],q[5];
x q[5];
rz(3*pi/4) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[8],q[4];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[8],q[4];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[3],q[2];
rz(-pi/4) q[8];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[12],q[13];
rz(pi/4) q[13];
cx q[8],q[13];
rz(-pi/4) q[13];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
rz(3*pi/4) q[12];
sx q[12];
rz(pi/2) q[12];
rz(pi/4) q[8];
cx q[13],q[8];
rz(pi/4) q[13];
rz(-pi/4) q[8];
cx q[13],q[8];
cx q[12],q[13];
x q[8];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[12],q[13];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[14],q[13];
rz(-pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[11],q[12];
rz(pi/4) q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
rz(3*pi/4) q[11];
sx q[11];
rz(pi/2) q[11];
rz(pi/4) q[13];
cx q[12],q[13];
rz(pi/4) q[12];
rz(-pi/4) q[13];
cx q[12],q[13];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
x q[13];
cx q[12],q[13];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[15],q[19];
cx q[19],q[15];
cx q[15],q[19];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[8],q[13];
rz(-pi/4) q[13];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[5],q[4];
rz(pi/4) q[4];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
rz(pi/4) q[13];
cx q[5],q[4];
cx q[4],q[5];
cx q[5],q[4];
rz(-pi/4) q[8];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
rz(3*pi/4) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[8],q[13];
rz(-pi/4) q[13];
rz(pi/4) q[8];
cx q[8],q[13];
x q[13];
cx q[4],q[8];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
rz(pi/2) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[4],q[8];
cx q[4],q[3];
cx q[3],q[4];
cx q[4],q[3];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[9],q[10];
rz(-pi/4) q[10];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[7],q[9];
rz(pi/4) q[9];
cx q[10],q[9];
rz(pi/4) q[10];
rz(-pi/4) q[9];
cx q[7],q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
rz(3*pi/4) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[9],q[10];
rz(-pi/4) q[10];
rz(pi/4) q[9];
cx q[9],q[10];
x q[10];
cx q[7],q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
cx q[9],q[10];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[10],q[9];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
cx q[1],q[0];
cx q[0],q[1];
cx q[1],q[0];
cx q[9],q[10];
cx q[10],q[9];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[15],q[14];
rz(pi/4) q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[12],q[13];
rz(pi/4) q[12];
rz(-pi/4) q[13];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
cx q[11],q[12];
cx q[12],q[11];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
rz(-pi/4) q[13];
rz(pi/4) q[14];
cx q[14],q[13];
x q[13];
rz(3*pi/4) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
rz(pi/2) q[15];
sx q[15];
rz(pi/2) q[15];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[17],q[15];
rz(-pi/4) q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[13],q[8];
cx q[17],q[15];
cx q[15],q[17];
cx q[17],q[15];
cx q[15],q[14];
cx q[14],q[15];
cx q[15],q[14];
cx q[8],q[13];
cx q[13],q[8];
cx q[4],q[8];
rz(pi/4) q[8];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[14],q[13];
rz(-pi/4) q[13];
cx q[13],q[8];
rz(pi/4) q[14];
cx q[8],q[13];
cx q[13],q[8];
cx q[14],q[13];
cx q[13],q[14];
cx q[14],q[13];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
rz(3*pi/4) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[8],q[13];
rz(-pi/4) q[13];
rz(pi/4) q[8];
cx q[8],q[13];
x q[13];
cx q[4],q[8];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
cx q[8],q[13];
cx q[8],q[4];
cx q[4],q[8];
cx q[8],q[4];
cx q[13],q[8];
cx q[8],q[13];
cx q[13],q[8];
cx q[12],q[13];
rz(-pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[11],q[12];
rz(pi/4) q[12];
cx q[13],q[12];
rz(-pi/4) q[12];
cx q[11],q[12];
rz(pi/4) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[11],q[12];
rz(pi/4) q[11];
rz(-pi/4) q[12];
cx q[11],q[12];
rz(3*pi/4) q[13];
sx q[13];
rz(pi/2) q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[13],q[12];
cx q[12],q[11];
rz(pi/2) q[11];
sx q[11];
rz(pi/2) q[11];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[10],q[9];
x q[13];
cx q[12],q[13];
cx q[9],q[10];
cx q[10],q[9];
cx q[7],q[9];
rz(-pi/4) q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[0],q[7];
rz(pi/4) q[7];
cx q[9],q[7];
rz(-pi/4) q[7];
cx q[0],q[7];
cx q[7],q[0];
cx q[0],q[7];
cx q[7],q[0];
rz(3*pi/4) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[9];
cx q[7],q[9];
rz(pi/4) q[7];
rz(-pi/4) q[9];
cx q[7],q[9];
cx q[0],q[7];
x q[9];
cx q[9],q[7];
cx q[7],q[9];
cx q[9],q[7];
cx q[0],q[7];
rz(pi/2) q[9];
sx q[9];
rz(pi/2) q[9];
cx q[10],q[9];
cx q[9],q[10];
cx q[10],q[9];
cx q[11],q[10];
cx q[10],q[11];
cx q[11],q[10];
cx q[16],q[11];
cx q[11],q[16];
cx q[16],q[11];
cx q[22],q[16];
rz(-pi/4) q[16];
cx q[22],q[16];
cx q[16],q[22];
cx q[22],q[16];
cx q[23],q[22];
rz(pi/4) q[22];
cx q[16],q[22];
rz(pi/4) q[16];
rz(-pi/4) q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[16];
rz(-pi/4) q[16];
rz(pi/4) q[22];
cx q[22],q[16];
x q[16];
rz(3*pi/4) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[23],q[22];
cx q[22],q[16];
cx q[16],q[22];
cx q[22],q[16];
rz(pi/2) q[16];
sx q[16];
rz(pi/2) q[16];
cx q[23],q[22];
cx q[22],q[16];
cx q[16],q[22];
cx q[22],q[16];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
cx q[24],q[23];
rz(-pi/4) q[23];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[25],q[24];
rz(pi/4) q[24];
cx q[23],q[24];
rz(pi/4) q[23];
rz(-pi/4) q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[24],q[23];
rz(-pi/4) q[23];
rz(pi/4) q[24];
cx q[24],q[23];
x q[23];
rz(3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[25],q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[26],q[25];
rz(-pi/4) q[25];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[38],q[26];
cx q[26],q[25];
rz(pi/4) q[25];
cx q[38],q[26];
cx q[26],q[25];
rz(-pi/4) q[25];
rz(pi/4) q[26];
cx q[38],q[26];
cx q[26],q[25];
rz(3*pi/4) q[25];
sx q[25];
rz(pi/2) q[25];
rz(pi/4) q[26];
rz(-pi/4) q[38];
cx q[26],q[38];
cx q[25],q[26];
x q[38];
cx q[26],q[38];
cx q[38],q[26];
cx q[26],q[38];
cx q[25],q[26];
rz(pi/2) q[38];
sx q[38];
rz(pi/2) q[38];
cx q[26],q[38];
cx q[38],q[26];
cx q[26],q[38];
cx q[26],q[25];
cx q[25],q[26];
cx q[26],q[25];
cx q[25],q[24];
cx q[24],q[25];
cx q[25],q[24];
cx q[23],q[24];
rz(-pi/4) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[21],q[22];
rz(pi/4) q[22];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[24],q[23];
rz(-pi/4) q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
rz(3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[22],q[23];
rz(pi/4) q[22];
rz(-pi/4) q[23];
cx q[22],q[23];
cx q[21],q[22];
x q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[21],q[22];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
rz(pi/2) q[23];
sx q[23];
rz(pi/2) q[23];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[28],q[24];
rz(-pi/4) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[21],q[22];
rz(pi/4) q[22];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[28],q[24];
cx q[24],q[28];
cx q[28],q[24];
cx q[24],q[23];
rz(-pi/4) q[23];
cx q[23],q[22];
cx q[22],q[23];
cx q[23],q[22];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
rz(3*pi/4) q[21];
sx q[21];
rz(pi/2) q[21];
rz(pi/4) q[24];
cx q[24],q[23];
cx q[23],q[24];
cx q[24],q[23];
cx q[22],q[23];
rz(pi/4) q[22];
rz(-pi/4) q[23];
cx q[22],q[23];
cx q[21],q[22];
cx q[22],q[21];
cx q[21],q[22];
cx q[22],q[21];
rz(pi/2) q[21];
sx q[21];
rz(pi/2) q[21];
cx q[21],q[20];
cx q[20],q[21];
cx q[21],q[20];
x q[23];
cx q[22],q[23];
cx q[27],q[20];
cx q[20],q[27];
cx q[27],q[20];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[30],q[29];
rz(-pi/4) q[29];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[31],q[30];
rz(pi/4) q[30];
cx q[29],q[30];
rz(pi/4) q[29];
rz(-pi/4) q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[29];
rz(-pi/4) q[29];
rz(pi/4) q[30];
cx q[30],q[29];
x q[29];
rz(3*pi/4) q[31];
sx q[31];
rz(pi/2) q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
rz(pi/2) q[29];
sx q[29];
rz(pi/2) q[29];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[29],q[27];
cx q[27],q[29];
cx q[29],q[27];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[30],q[29];
cx q[29],q[30];
cx q[30],q[29];
cx q[32],q[31];
rz(-pi/4) q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[33],q[32];
rz(pi/4) q[32];
cx q[31],q[32];
rz(pi/4) q[31];
rz(-pi/4) q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
rz(-pi/4) q[31];
rz(pi/4) q[32];
cx q[32],q[31];
x q[31];
rz(3*pi/4) q[33];
sx q[33];
rz(pi/2) q[33];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
rz(pi/2) q[33];
sx q[33];
rz(pi/2) q[33];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[30],q[31];
rz(-pi/4) q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[32];
rz(pi/4) q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[30],q[31];
rz(pi/4) q[30];
rz(-pi/4) q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
cx q[31],q[30];
cx q[30],q[31];
cx q[31],q[30];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
rz(-pi/4) q[31];
rz(pi/4) q[32];
cx q[32],q[31];
x q[31];
rz(3*pi/4) q[33];
sx q[33];
rz(pi/2) q[33];
cx q[32],q[33];
cx q[33],q[32];
cx q[32],q[31];
rz(pi/2) q[33];
sx q[33];
rz(pi/2) q[33];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[35],q[34];
rz(-pi/4) q[34];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[39],q[35];
cx q[35],q[34];
rz(pi/4) q[34];
cx q[39],q[35];
cx q[35],q[34];
rz(-pi/4) q[34];
rz(pi/4) q[35];
cx q[39],q[35];
cx q[35],q[34];
rz(3*pi/4) q[34];
sx q[34];
rz(pi/2) q[34];
rz(pi/4) q[35];
rz(-pi/4) q[39];
cx q[35],q[39];
cx q[34],q[35];
cx q[35],q[34];
cx q[34],q[35];
cx q[35],q[34];
rz(pi/2) q[34];
sx q[34];
rz(pi/2) q[34];
cx q[34],q[33];
cx q[33],q[34];
cx q[34],q[33];
cx q[33],q[32];
cx q[32],q[33];
cx q[33],q[32];
cx q[31],q[32];
rz(-pi/4) q[32];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
x q[39];
cx q[35],q[39];
cx q[42],q[36];
cx q[36],q[42];
cx q[42],q[36];
cx q[36],q[31];
rz(pi/4) q[31];
cx q[32],q[31];
rz(-pi/4) q[31];
rz(pi/4) q[32];
cx q[36],q[31];
cx q[32],q[31];
cx q[31],q[32];
cx q[32],q[31];
rz(3*pi/4) q[32];
sx q[32];
rz(pi/2) q[32];
cx q[36],q[31];
rz(-pi/4) q[31];
rz(pi/4) q[36];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
x q[36];
cx q[36],q[31];
cx q[31],q[36];
cx q[36],q[31];
cx q[32],q[31];
rz(pi/2) q[36];
sx q[36];
rz(pi/2) q[36];
cx q[42],q[36];
cx q[36],q[42];
cx q[42],q[36];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[45],q[44];
rz(-pi/4) q[44];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[46],q[45];
rz(pi/4) q[45];
cx q[44],q[45];
rz(pi/4) q[44];
rz(-pi/4) q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
rz(-pi/4) q[44];
rz(pi/4) q[45];
cx q[45],q[44];
x q[44];
rz(3*pi/4) q[46];
sx q[46];
rz(pi/2) q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(pi/2) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[43],q[44];
rz(-pi/4) q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[42],q[43];
rz(pi/4) q[43];
cx q[44],q[43];
rz(-pi/4) q[43];
cx q[42],q[43];
rz(pi/4) q[44];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[42],q[43];
rz(pi/4) q[42];
rz(-pi/4) q[43];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
x q[42];
rz(3*pi/4) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
rz(pi/2) q[42];
sx q[42];
rz(pi/2) q[42];
cx q[44],q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
cx q[41],q[42];
cx q[42],q[41];
cx q[41],q[40];
cx q[40],q[41];
cx q[41],q[40];
cx q[44],q[43];
cx q[43],q[44];
cx q[44],q[43];
cx q[46],q[45];
cx q[46],q[37];
cx q[37],q[46];
cx q[46],q[37];
cx q[46],q[45];
cx q[45],q[46];
cx q[46],q[45];
cx q[45],q[44];
rz(-pi/4) q[44];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
cx q[46],q[58];
cx q[58],q[46];
cx q[46],q[58];
cx q[46],q[45];
rz(pi/4) q[45];
cx q[44],q[45];
rz(pi/4) q[44];
rz(-pi/4) q[45];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
cx q[45],q[44];
rz(3*pi/4) q[44];
sx q[44];
rz(pi/2) q[44];
cx q[46],q[45];
rz(-pi/4) q[45];
rz(pi/4) q[46];
cx q[46],q[45];
cx q[45],q[44];
cx q[44],q[45];
x q[44];
cx q[45],q[46];
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
cx q[42],q[43];
rz(-pi/4) q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[41],q[42];
rz(pi/4) q[42];
cx q[43],q[42];
rz(-pi/4) q[42];
cx q[41],q[42];
rz(pi/4) q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[41],q[42];
rz(pi/4) q[41];
rz(-pi/4) q[42];
cx q[41],q[42];
rz(3*pi/4) q[43];
sx q[43];
rz(pi/2) q[43];
cx q[43],q[42];
cx q[42],q[43];
cx q[43],q[42];
cx q[42],q[41];
x q[43];
cx q[42],q[43];
