function  [CorrectRate,Correct,SolveTime,Wu,Bu]  = Main_DC(MSVTASolver,data,Y,test_X,test_Y,TestNum)
 
%% Step1 set parameters
 
    SolveTime = 0 ;

    DsNum  = MSVTASolver.DsNum ;  
    T  = MSVTASolver.T  ;  
    L  = MSVTASolver.L ;  
    N  = MSVTASolver.N  ;  
    C  = MSVTASolver.C  ;  
   
    lambdaT  = MSVTASolver.lambdaT ;      
    gamma     = MSVTASolver.gamma ;  

    KwType  = MSVTASolver.KwType ;
    KwValue = MSVTASolver.KwValue ;

    Task = MSVTASolver.Task ; 
   
  
%% Step2 Extend Kernel 
    PrepareKernel = clock;
   
    [Minv,Bpinv,P]  =DC_Minv_P(T,L,N,lambdaT,gamma); 
      
    PMP = P' * Minv * P   ;   
    PBP = P' * Bpinv * P    ; 
    YY = Y*Y' ;

    K = Mygram(data,data,KwType,KwValue) ;  

    CommonH = PMP.*K.*YY + PBP.*YY +  eye(N)./C;  
       

  
    Kernel_OK = clock;     KernelTime = etime(Kernel_OK,PrepareKernel) ;
    disp(['KernelTime=',num2str(KernelTime)])  ;

%% Step3 
    
    GISVM_Start = clock;  
    
    [alpha,support] = Dual_iqph(CommonH,Y,C); 
    GISVM_OK = clock;     SolveTime = etime(GISVM_OK,GISVM_Start) ;   
    disp(['DC_SVM_Time=',num2str(SolveTime)])  ;    
 
  %% Prepare the Test 
  % G.L. Grinblat, L.C. Uzal, H.A. Ceccatto, P.M. Granitto, Solving nonstationary classification problems with coupled support vector machines, IEEE Transactions on Neural Networks, 22 (2010) 37-51.
    PrepareTest_Start = clock;  
    support = find(alpha > 1e-6);  
    alpha = alpha(support,:);
    Y = Y(support,:) ;
    P = P(:,support) ;
    data = data(support,:) ;
    BB =  alpha  .* Y  ;
    Bu = Bpinv * P  * BB; 
    Prepare_OK = clock;     PrepareTest_Time = etime(Prepare_OK,PrepareTest_Start) ;
    NTang_Test_Start = clock; 
    CorrectRate=0;
    Correct=zeros(T,DsNum) ;
    for num = 1:TestNum
        Start = (num-1)*N;
        Testdata =  test_X( Start+1:Start+N , : ) ; 
        TestY = test_Y ( Start+1:Start+N , : ) ;
        GG = Mygram(data,Testdata,KwType,KwValue ) ;                 
        Wu = Minv * P * diag(alpha  .* Y ) * GG ;  
        TestErrors = 0;
        TestRights = 0;
        if ( strcmp(Task, 'Pridict' ) ) 
            m = 1 ;
        end 
        for t= 1:T  
            for k=1:DsNum
                
                w2 = Wu( (t-1) * DsNum +k ,:);       
                b2 = Bu( (t-1)* DsNum +k ,1) ;
  
                for j=1:L
                    Result = w2 ( (t-1)* DsNum *L + (k-1)*L +j )   + b2;

                    if(sign(Result)~=TestY( (t-1)* DsNum *L + (k-1)*L +j ,1) )
                       TestErrors  = TestErrors  +1;                      
                    end

                    if(sign(Result)==TestY( (t-1)* DsNum *L + (k-1)*L +j ,1) )
                        TestRights = TestRights +1;
                        Correct(t,k)=Correct(t,k)+1 ;
                    end         
                 end 
             end
         
        end        
        CorrectRate = CorrectRate + sum(TestRights)  ;      
    end
    
    N_Test_OK = clock;     NTest_Time = etime(N_Test_OK,N_Test_Start) ;
    disp(['NTangTest_Time=',num2str(NTest_Time)])  ;

    CorrectRate = CorrectRate /(T* DsNum *L*TestNum);
    disp(['    CorrectRate = ',num2str(CorrectRate)]); % disp('   ');
    disp(['TestRights = ',num2str(TestRights)]);  disp('   ');
    disp(['TestErrors = ',num2str(TestErrors)]);  disp('   ');

end  

 





