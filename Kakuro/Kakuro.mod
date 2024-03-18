int a = ...;
int b = ...;
int n = ...;
range range_a = 1..a;
range range_b = 1..b;
range range_n = 1..n;

int E[range_a][range_b] = ...;
int T[range_n][1..4] = ...;
int L[range_n] = ...;

dvar int W[range_a][range_b];

subject to{
   //W[i][j] when E[i][j] == 0 must be between 1 and 9
   forall (i in range_a, j in range_b : E[i, j] == 0){
     W[i, j] >= 1;
     W[i, j] <= 9;
   }

   //forall triangles summing by row
   forall (i in range_n : T[i, 3] == 0){
     //sum of numbers moving by column is equal to sum in triangle
     sum (j in (T[i, 2]+1)..(T[i, 2]+L[i]))W[T[i, 1], j] == T[i, 4];
     //numbers in sum moving by column are non-repeating
     forall (j in (T[i, 2]+1)..(T[i, 2]+L[i]), k in (T[i, 2]+1)..(T[i, 2]+L[i]) : j != k)
       W[T[i, 1], j] != W[T[i, 1], k];}

   //forall triangles summing by column
   forall (i in range_n : T[i, 3] == 1){
     //sum of numbers moving by row is equal to sum in triangle
     sum (j in (T[i, 1]+1)..(T[i, 1]+L[i]))W[j, T[i, 2]] == T[i, 4];
     //numbers in sum moving by row are non-repeating
     forall (j in (T[i, 1]+1)..(T[i, 1]+L[i]), k in (T[i, 1]+1)..(T[i, 1]+L[i]) : j != k)
       W[j, T[i, 2]] != W[k, T[i, 2]];}

   //set W[i, j] to 0 where E[i, j] == 1
   forall (i in range_a, j in range_b : E[i, j] == 1)
     W[i, j] == 0;
}
