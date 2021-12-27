#include <vector>
#include <iostream>

using namespace std;

int main(){

    vector<vector< double >> test = {{1, 2}, {2, 3, 4}}; 

    for(int i = 0; i < test.size(); i++){
        for(int j = 0; j < test[i].size(); ++j){
            cout << test[i][j] << " ";
        }
        cout << endl; 
    }

}