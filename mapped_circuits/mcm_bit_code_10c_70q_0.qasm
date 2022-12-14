OPENQASM 2.0;
include "qelib1.inc";
qreg q[70];
creg m_mcm0[27];
creg m_meas_all[55];
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
