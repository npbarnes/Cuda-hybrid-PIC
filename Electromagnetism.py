import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from string import Template
class PIC:

    def __init__(self, numParts, E, B, step, wx=1, wy=1, wz=1):
        self.numParts = numParts
        f = open("cudaTools.cu","r")
        cuTools = f.read()
        f.close()
        self.subs = {   'Ex':E.x, 'Ey':E.y, 'Ez':E.z,
                        'Bx':B.x, 'By':B.y, 'Bz':B.z, 
                        'wx':wx, 'wy':wy, 'wz':wz, 
                        'step':step, 'numParts':numParts, 'cudaTools':cuTools } 

        self.mod_template = Template("""//CUDA//
            ${cudaTools}

            __device__ void boris(float* v)
            {
                float vmx = v[0] + (${step}/2) * ${Ex};
                float vmy = v[1] + (${step}/2) * ${Ey};
                float vmz = v[2] + (${step}/2) * ${Ez};

                float tx = (${step}/2) * ${Bx};
                float ty = (${step}/2) * ${By};
                float tz = (${step}/2) * ${Bz};

                float vprx = vmx + (vmy*tz - vmz*ty);
                float vpry = vmy + (vmz*tx - vmx*tz);
                float vprz = vmz + (vmx*ty - vmy*tx);

                float sx = (2/(1+pow(tx,2)+pow(ty,2)+pow(tz,2))) * tx;
                float sy = (2/(1+pow(tx,2)+pow(ty,2)+pow(tz,2))) * ty;
                float sz = (2/(1+pow(tx,2)+pow(ty,2)+pow(tz,2))) * tz;

                float vpx = vmx + (vpry*sz - vprz*sy);
                float vpy = vmy + (vprz*sx - vprx*sz);
                float vpz = vmz + (vprx*sy - vpry*sx);

                v[0] = vpx + (${step}/2) * ${Ex};
                v[1] = vpy + (${step}/2) * ${Ey};
                v[2] = vpz + (${step}/2) * ${Ez};
            }

            __device__ void euler(float* x, float* v)
            {
                x[0] = x[0] + ${step}*v[0];
                x[1] = x[1] + ${step}*v[1];
                x[2] = x[2] + ${step}*v[2];
            }

            __device__ void periodic(float* x)
            {
                x[0] = x[0] - ${wx}*floorf(x[0]/${wx});
                x[1] = x[1] - ${wy}*floorf(x[1]/${wy});
                x[2] = x[2] - ${wz}*floorf(x[2]/${wz});
            }

            __global__ void multiBoris(float* x, float* v, int n)
            {
                int id = getGlobalId();
                int tid = getThreadId();
                if(id < ${numParts})
                {
                    // Setup cache.
                    // The first index specifies the thread (within a block, ie 0-1023)
                    // the second index specifies x or v
                    // the thrid index is the component.
                    __shared__ float cache[1024][2][3];

                    // load cache
                    cache[tid][0][0] = x[index(id,0)];
                    cache[tid][0][1] = x[index(id,1)];
                    cache[tid][0][2] = x[index(id,2)];

                    cache[tid][1][0] = v[index(id,0)];
                    cache[tid][1][1] = v[index(id,1)];
                    cache[tid][1][2] = v[index(id,2)];

                    for(int i=0;i<n;i++)
                    {
                        // Update velocities
                        boris(cache[tid][1]);

                        // Move particles
                        euler(cache[tid][0],cache[tid][1]);

                        // Periodic boundary conditions
                        periodic(cache[tid][0]);
                    }

                    // unload cache
                    x[index(id,0)] = cache[tid][0][0];
                    x[index(id,1)] = cache[tid][0][1];
                    x[index(id,2)] = cache[tid][0][2];

                    v[index(id,0)] = cache[tid][0][0];
                    v[index(id,1)] = cache[tid][0][1];
                    v[index(id,2)] = cache[tid][0][2];

                }
            }
            """)
        self.mod = SourceModule(self.mod_template.substitute(self.subs))

        self.cuBoris = self.mod.get_function('multiBoris')

    def stepN(self,positions,velocities,n):
        x_gpu = cuda.mem_alloc(positions.nbytes)
        v_gpu = cuda.mem_alloc(velocities.nbytes)

        cuda.memcpy_htod(x_gpu,positions)
        cuda.memcpy_htod(v_gpu,velocities)

        import numpy as np
        self.cuBoris(x_gpu, v_gpu, np.int32(n), block=(1024,1,1), grid=(self.numParts/1024 + 1,1))

        cuda.memcpy_dtoh(positions,x_gpu)
        cuda.memcpy_dtoh(velocities,v_gpu)

