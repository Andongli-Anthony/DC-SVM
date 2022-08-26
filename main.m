clear;clc;

DataStr = 'Testdata' ; 
load(DataStr);

T= 'set the number of sub-datasets'; 
K= 'set the number of Tasks';  
L='set the number of samples in each sub-dataset'; 
N= T*K*L; 
C = 'set Kernel parameter'   ;
method = 'DC_SVM_Linear';

iStart = 1 ;   iStep = 1 ; iEnd = 21;
kStart = 1;  kStep = 1 ; kEnd =21;  
TrainGroup = 10;     
Group = 10; 
Testnum= 10 ;  

%% Prepare data
TrainData =[]; TrainY=[];  TestData =[]; TestY=[]; ValidData =[]; ValidY=[];
for group=1:Group  
    S1  = (group-1)*(T*L) + 0;     
    S2  = (group-1)*(T*L) +  (T*L)*Testnum;  
    S3  = (group-1)*(T*L)+  (T*L)*Testnum*2;  
    for j=1:T  
        
        TrainData = [TrainData ; T1_X(S1+j:S1+j+L-1 ,:); T2_X(S1+j:S1+j+L-1 ,:)  ];
        TrainY =    [TrainY ;    T1_Y(S1+j:S1+j+L-1 ,:); T2_Y(S1+j:S1+j+L-1 ,:)  ];

        TestData = [TestData ;   T1_X(S2+j:S2+j+L-1 ,:); T2_X(S2+j:S2+j+L-1 ,:)  ];
        TestY =    [TestY;       T1_Y(S2+j:S2+j+L-1 ,:); T2_Y(S2+j:S2+j+L-1 ,:)  ];

        ValidData = [ValidData ; T1_X(S3+j:S3+j+L-1 ,:); T2_X(S3+j:S3+j+L-1 ,:)  ];
        ValidY =    [ValidY;     T1_Y(S3+j:S3+j+L-1 ,:); T2_Y(S3+j:S3+j+L-1 ,:)  ];
    end
end

MSVTASolver.Fun= 'DC_SVM' ;
MSVTASolver.CalKernel= 'DCouple' ;                  
MSVTASolver.KwType= 'linear';              

%% Set parameters
MSVTASolver.DsNum=K ;            
MSVTASolver.T=T ;              
MSVTASolver.L=L ;               
MSVTASolver.N=N ;             
MSVTASolver.C= C ;         
MSVTASolver.KwValue=1;        

MSVTASolver.Task = 'Estimate' ; 
MSVTASolver.MEB_eps = 1e-6 ;

lambda_T = 2.^[ 0:1:20 ];  
gamma = 2.^[ 0:1:20 ];  

TotalCorrect=zeros(T,K);
Result =zeros(T*Group,K);
BestPram = zeros(Group,7);
CRtest  = zeros(21,21); 
AllGroup= zeros(21,21,Group);
SubTaskCorrect=zeros(Group,K);

for group = 1: TrainGroup
    %% Step1 All Prameter
    CRtest  = AllGroup(:,:,group);    
    grp_start = clock;    
    trainX= TrainData((group-1)*N+1:(group-1)*N+N,:);
    trainY= TrainY((group-1)*N+1:(group-1)*N+N,:);
    for i =  iStart : iStep : iEnd   
        MSVTASolver.lambdaT=lambda_T(i);
        for k =  kStart: kStep :kEnd
            MSVTASolver.gamma= gamma(k);
            [CRtest(i,k) ] = Main_DC( MSVTASolver,trainX,trainY,ValidData,ValidY,Testnum); 
           AllGroup(i,k,group)= CRtest(i,k);
        end
        disp( '  ');
    end
    grp_end = clock;   grp_Time = etime(grp_end,grp_start) ;
    disp(['Group_',num2str(group),'  Time=',num2str(grp_Time),'......'])  ;
    
    %% Step2 Find Best Prameter 
    [BestValue,I] = max(CRtest(:));
    [r1,c1] = find(BestValue == CRtest);
    Row = r1(1) ; Column = c1(1);
      
    BestPram( group , 1) = MSVTASolver.C ;
    BestPram( group , 2) = lambda_T(Row) ;
    BestPram( group , 3) = gamma(Column)  ;

    BestPram( group , 5) = BestValue ;
    disp(['BestValue=',num2str(BestValue)])  ;
    
     %% Step3 Pridict & Estimate
    MSVTASolver.lambdaT = BestPram( group , 2)  ;
    MSVTASolver.gamma = BestPram( group , 3)   ; 

    [ValidValue,SubCorrect,SolveTime] = Main_DC( MSVTASolver,trainX,trainY,TestData,TestY,Testnum);
    BestPram( group , 6) = ValidValue ; 
    BestPram( group ,7) = SolveTime ; 
    TotalCorrect = TotalCorrect + SubCorrect ;
    SubTaskCorrect(group,:)=sum(SubCorrect);
    disp(['ValidValue=',num2str(ValidValue)])  ; disp('   '); disp('   ');
  
end