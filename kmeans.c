#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "stdlib.h"
#include "stdio.h"

typedef struct Cluster {
    int name, size;
    double *centroid;
    double *sum_of_obs;
} Cluster;

typedef struct Observation {
    double *values;
    Cluster *cluster;
} Observation;

static PyObject* run (PyObject *self, PyObject *args);
static void print_index(PyObject *index, int K);

static int kmeans(PyObject *observations, int k, int n, int d, int max_iter);

static void convert_obs(Observation **input_values, PyObject *observations, int N, int d);

static void clean(Observation **observations, int n, Cluster **cluster_array, int k);

static int init(Observation **observations, int n, int d);

static void find_closest_cluster(Observation *observation, Cluster **clusters_array, int k, int d);

static double squared_euclidean_distance(double *observation, double *centroid, int d);

static void observation_sum(double *sum_of_obs, double *observation_values, int d);

static void observation_sub(double *sum_of_obs, const double *observation_values, int d);

static void remove_obs(Observation *observation,int d) ;

static int update_centroid(Cluster **clusters_array, int k, int d);

static void insert_obs(Observation *observation, Cluster *best_cluster, int d);

static int create_k_clusters(Observation **observations, Cluster **clusters_array, int k, int d);

static void print_clusters(Cluster **clusters_array, int k, int d);

static void copy(const double *values, double *sum_of_obs, int d);

static int kmeans(PyObject *observations, int k, int n, int d, int max_iter) {
    Observation **input_values;
    int is_changed_from_last_run, found_k_clusters, number_of_iter,obs_num;
    Cluster **clusters_array;

    input_values=malloc(n*sizeof(Observation));
    if (input_values==NULL){
        return -1;
    }

    if (init(input_values, n, d)==-1){
        return -1;}
    convert_obs(input_values, observations, n, d);

    is_changed_from_last_run= 1;
    found_k_clusters = 0;
    number_of_iter = 1;
    obs_num = k;

    clusters_array=malloc(k*sizeof(Cluster));
    if (clusters_array==NULL){
        return -1;
        }

    while (is_changed_from_last_run == 1 && (number_of_iter <= max_iter)) {
        /*main loop*/
        if (found_k_clusters == 1) {
            /*k cluster have been initiated*/
            find_closest_cluster(input_values[obs_num], clusters_array, k, d);
            obs_num+=1;

        } else {
            /*initiate k clusters*/
            if (create_k_clusters(input_values, clusters_array, k, d)==-1){
            return -1;
            }
            found_k_clusters = 1;
        }
        if (obs_num == n) {
            /*start new iteration*/
            is_changed_from_last_run=update_centroid(clusters_array,k,d);
            obs_num = 0;
            number_of_iter += 1;


        }
    }
    print_clusters(clusters_array, k, d);
    clean(input_values, n, clusters_array, k);
    return 0;
}

static void print_clusters(Cluster **clusters_array, int k, int d) {
    int i;
    for (i = 0; i < k; i++) {
        int j;
        for (j = 0; j < d; j++) {
            if (j < d - 1) {
                printf("%f%s", clusters_array[i]->centroid[j], ",");
            } else {
                printf("%f", clusters_array[i]->centroid[j]);
            }
        }
        if (i != k - 1) {
            printf("\n");
        }
    }
}

static int create_k_clusters(Observation **observations, Cluster **clusters_array, int k, int d) {
    int index;
    for (index = 0; index < k; index++) {
        clusters_array[index] = malloc(sizeof(Cluster));
        if (clusters_array[index]==NULL){ return -1; }
        clusters_array[index]->name = index;
        clusters_array[index]->size = 1;
        clusters_array[index]->centroid = calloc(d, sizeof(double ));
        if (clusters_array[index]->centroid==NULL){ return -1; }
        copy(observations[index]->values, clusters_array[index]->centroid, d);
        clusters_array[index]->sum_of_obs = calloc(d, sizeof(double ));
        if (clusters_array[index]->sum_of_obs ==NULL){
            return -1;
        }
        copy(observations[index]->values, clusters_array[index]->sum_of_obs, d);
        observations[index]->cluster=clusters_array[index];
    }
    return 0;
}

static void copy(const double *values, double *sum_of_obs, int d) {
    int i;
    for (i=0; i<d; i++){
        sum_of_obs[i]=values[i];
    }

}

static int init(Observation **observations, int n, int d) {
    int i;
    for (i = 0; i < n; i++) {
        observations[i] = malloc(sizeof(Observation));
        if (observations[i]==NULL){
            return -1;
        }
        observations[i]->values = (double *) calloc(d, sizeof(double));
        if (observations[i]->values==NULL){
            return -1;
            }
        observations[i]->cluster = NULL;
    }
    return 0;
}

static void convert_obs(Observation **input_values, PyObject *observations, int n, int d){
    /*
    * convert the PyObject type observations to arrys of type double
    */
    int i, j;
    PyObject *obs, *val;
    Py_ssize_t obs_num, obs_size;
    obs_num= PyList_Size(observations);
    for (i=0; i<obs_num; i++){
        obs=PyList_GetItem(observations, i);
        if (!PyList_Check(obs)){
           return;
        }
        obs_size=PyList_Size(obs);
        for (j=0; j<obs_size; j++){
            val=PyList_GetItem(obs, j);
            if (!PyFloat_Check(val)){
                return;
                }
            input_values[i]->values[j]=PyFloat_AsDouble(val);
            if (input_values[i]->values[j]== -1 && PyErr_Occurred()){
            /* double too big to fit in a C float, bail out */
                printf("error");
                return;
            }
        }
    }
}


static void clean(Observation **observations, int n, Cluster **cluster_array, int k) {
    int i,j;
    for (i = 0; i < n; i++) {
        free(observations[i]->values);
        free(observations[i]);
    }
    free(observations);

    for (j = 0; j < k; j++) {
        free(cluster_array[j]->sum_of_obs);
        free(cluster_array[j]->centroid);
        free(cluster_array[j]);
    }
    free(cluster_array);
}


static void find_closest_cluster(Observation *observation, Cluster **clusters_array, int k, int d) {
    /*find closest cluster for observation (of class Observation)
    size of clusters_array is K, each index is of struct Cluster */

    int index;
    double min_dist;
    double dist;
    Cluster *best_cluster=NULL;

    min_dist = squared_euclidean_distance(observation->values, clusters_array[0]->centroid, d);
    best_cluster = clusters_array[0];

    for (index=1; index < k; index++) {
        dist = squared_euclidean_distance(observation->values, clusters_array[index]->centroid, d);
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = clusters_array[index];
        }
    }
    if (observation->cluster != NULL){
        if (observation->cluster->name == best_cluster->name){ return; }
        remove_obs(observation, d);
    }
    insert_obs(observation, best_cluster, d);
}

static double squared_euclidean_distance(double *observation, double *centroid, int d){
    /*find cluster’s centroid using squared Euclidean distance
    observation and centroid are lists of size D*/
    int index;
    double dist;
    double temp;
    dist = 0;

    for (index =0; index < d; index++) {
        temp = (observation[index] - centroid[index]);
        dist += (temp*temp);
    }
    return dist;
}

static void observation_sum(double *sum_of_obs, double *observation_values, int d){
    /* sum_of_obs is a list in len D that sums all observations that belongs to the cluster*/
    int index;
    for (index=0; index<d; index++){
        sum_of_obs[index] += observation_values[index];
    }
}

static void insert_obs(Observation *observation, Cluster *best_cluster, int d) {
    observation_sum(best_cluster->sum_of_obs, observation->values, d);
    best_cluster->size++;
    observation->cluster = best_cluster;
}

static void observation_sub(double *sum_of_obs, const double *observation_values, int d) {
    /*update sum_of_obs sum_of_obs is a list in len D that sums all observations that belongs to the cluster*/
    int index;
    for (index=0; index < d; index++){
        *(sum_of_obs + index) -= *(observation_values + index);
    }
}

static void remove_obs(Observation *observation,int d) {
    observation->cluster->size -= 1;
    observation_sub(observation->cluster->sum_of_obs, observation->values, d);
}

static int update_centroid(Cluster **clusters_array, int k, int d){
    /*update centroid using the sum of observations that belongs to the cluster */
    int dpoint;
    int cluster_index;
    int is_changed;
    double temp_calc;
    is_changed = 0;

    for (cluster_index=0; cluster_index<k ;cluster_index++) {
        /*iterate over the clusters*/
        Cluster *current_cluster;
        current_cluster= clusters_array[cluster_index];
        for (dpoint=0; dpoint<d; dpoint++){
            temp_calc = current_cluster->sum_of_obs[dpoint]/(float)current_cluster->size;
            if (temp_calc != current_cluster->centroid[dpoint]) {
                /*check if the centroid in place dpoint should be updated*/
                current_cluster->centroid[dpoint] = temp_calc;
                is_changed = 1;
            }
        }
    }
    return is_changed;
}

static void print_index(PyObject *index, int K){
    /*print  the indexes of the the observations chosen to be clusters */
    int i;
    PyObject *ind;
    for (i=0; i<K-1; i++){
        ind=PyList_GetItem(index, i);
        printf("%ld%s", PyLong_AsLong(ind),",");
    }
    ind=PyList_GetItem(index, K-1);
    printf("%ld\n", PyLong_AsLong(ind));
}

static PyObject* run (PyObject *self, PyObject *args){
    /*
    * this function is the module's endpoint for communicating with python script
    * the function calculates kmeans algorithm and prints it to the requested file.
    * args:
        input_observation - matrix of observation where the first k observation are the clusters
        K - number of centroids required
        N - number of observations
        d - the dimension of each observation
        MAX - max iterations the script should do
        index - the indexes of the the observations chosen to be clusters
    * returnes None if the process ended successfully
    */

    int K,N,d,MAX_ITER;
    PyObject *input_observation, *index;
    if(!PyArg_ParseTuple(args, "(OiiiiO):run", &input_observation, &K, &N, &d, &MAX_ITER, &index)) {
        return PyErr_Format(PyExc_ValueError, "error in args");
    }
    if (!PyList_Check(input_observation)){
        return PyErr_Format(PyExc_ValueError, "not a list");
    }
     if (!PyList_Check(index)){
        return PyErr_Format(PyExc_ValueError, "not a list");
    }
    print_index(index, K);
    if (kmeans(input_observation, K, N, d, MAX_ITER)==-1){
        PyErr_Format(PyExc_ValueError, "Running Kmeans calculation using c module has failed");
    };
    Py_RETURN_NONE;
    }

static PyMethodDef capiMethods[] = {
    {"run",                   /* the Python method name that will be used */
      (PyCFunction)run, /* the C-function that implements the Python function and returns static PyObject*  */
      METH_VARARGS,           /* flags indicating parametersaccepted for this function */
      PyDoc_STR("kmeans++")}, /*  The docstring for the function */
    {NULL, NULL, 0, NULL}     /* The last entry must be all NULL as shown to act as a
                                 sentinel. Python looks for this entry to know that all
                                 of the functions for the module have been defined. */
};


/* This initiates the module using the above definitions. */
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,  /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    capiMethods /* the PyMethodDef array from before containing the methods of the extension */
};


/*
 * The PyModuleDef structure, in turn, must be passed to the interpreter in the module’s initialization function.
 * The initialization function must be named PyInit_name(), where name is the name of the module and should match
 * what we wrote in struct PyModuleDef.
 * This should be the only non-static item defined in the module file
 */
PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}