clear all; close all; clc; tic
% John Garofalo 
% ELE 559 - Signal Detection Theory
% Dr. Fonseca
% Graduate Project

% Using the Bayesian Optimum Classifier method, we can implement Projection
% Histograms as features for handwritten digit recognition to reduce the
% probability of error in detecting and identifiying digits a-priori with a
% trained classifier.

global PMFV PMFH PMFLV PMFLH DP DL PMFV_ PMFH_ PMFLV_ PMFLH_ DP_ DL_

% global nzEntries nzValues nzEntries_ nzValues_

totalDigits = 60000; trainingDigits = 55000;

DECISIONS = 0; ERRORS = 0;

% Collect data, and true digit information. Create a 3D array of amplitude and binary digits...

dataset = csvread( 'dataset.csv' ); digitNumber = dataset( :, 1 );

for k = 1:trainingDigits    
    digitMatrixCube( :, :, k) = reshape( dataset( k, 2 : 785 ), 28, 28 )';
end

% Four dimensional storage of digits...

DIGITS = zeros( 10, 28, 28, 7000 ); % Pre-allocate for speed...

for n = 1:10    
    j = 1; 
    for k = 1:trainingDigits    
        if ( digitNumber( k ) == ( n - 1 ) )            
            DIGITS( n, :, :, j ) = digitMatrixCube( :, :, k ); j = j + 1;        
        end
        N( n ) = j;
    end
    N( n ) = N( n ) - 1;
end
DIGITSV = ( DIGITS > 127 ); % The hist() function only works for columns, transpose middle digit dimensions...
DIGITSH = ( permute( DIGITS, [ 1, 3, 2, 4 ] ) > 127 ); % We need seperate DIGIT arrays to generate PMFs in a single nest cascade...

% It's time to train the new feature PMFs. Generate a-posteriori horizontal
% and vertical cumulative PMFs of pixel amplitudes for digits 0-9...

PMFV = zeros( 10, 28 ); PMFH = zeros( 10, 28 ); 
PMFV_ = zeros( 10, 28 ); PMFH_ = zeros( 10, 28 );
sPMF = zeros( 10, 1 ); D = zeros( 10, 28, 10);

for n = 1:10
    for k = 1:N( n )
        for i = 1:28
           PMFV_( n, i ) = PMFV_( n, i ) + size( find( DIGITSV( n, :, i, k ) ~= 0), 2 ); % Accumulate Histogram bin vaues ( pixel values) across all rows and columns...          
           PMFH_( n, i ) = PMFH_( n, i  ) + size( find( DIGITSH( n, :, i, k ) ~= 0), 2 );                     
           sPMF( n ) = sPMF( n ) + size( find( DIGITSV( n, :, :, k)  ~= 0 ), 1 ); % Accumulate the number of pixels per digit frame...      
        end
    end
    PMFV( n, :, : ) = PMFV_( n, :, : ) / sPMF( n ); PMFH( n, :, : ) = PMFH_( n, :, : ) / sPMF( n ); % Divide the total bin values per row and column per digit by the total number of pixels per digit...
end
PMFV = circshift( PMFV, -1, 1 ); PMFH = circshift( PMFH, -1, 1 ); % Circular shift to place the PMFs of zero at index ten.

% FEATURE #1: Rows with mean maximum number of pixels! Four per projection!

for i = 1:10
    for j = 1:4    
        [ maxPixelsAmplitudesV( i, j ), maxPixelIndexesV( i, j ) ] = max( PMFV( i, : ) ); 
        PMFV( i, maxPixelIndexesV( i, j ) ) = 0;    
         [ maxPixelsAmplitudesH( i, j ), maxPixelIndexesH( i, j ) ] = max( PMFH( i, : ) ); 
        PMFH( i, maxPixelIndexesH( i, j ) ) = 0;    
    end
end

% FEATURE #2: Rows with mean length values per max pixel count per digit! Four per projection!

BOUNDARYV_ = zeros( 28,1 ); BOUNDARYH_ = zeros( 28,1 ); BOUNDARYV = zeros( 1,1 ); BOUNDARYH = zeros( 1,1 ); 

rowLengthV = zeros( 10,28 ); rowLengthH = zeros( 10, 28 );
rowLengthsTotalV = zeros( 10,1 ); rowLengthsTotalH = zeros( 10, 1 );

PMFLV = zeros( 10, 28 ); PMFLH = zeros( 10, 28 );

a = 1; b = 1; q = 1; p = 1;
for n = 1:10    
    for k = 1:N(n)        
        for i = 1:28               
            for j = 1:28
                
                % Find digit length per row.
                
                if ( DIGITSV( n, i, j, k) == 1) 
                    BOUNDARYV_( a ) = j; a = a + 1; % Locate where digit existence begins and ends.
                elseif ( DIGITSH( n, i ,j ,k ) == 1)
                    BOUNDARYH_( b ) = j; b = b + 1;
                end     
                
                % Find digit length per row. Locate where digit existence begins and ends.
                
                for z = 1:28                      
                    if( BOUNDARYV_( z ) ~= 0 )                    
                            BOUNDARYV( q ) = BOUNDARYV_( z ); q = q + 1; % Get rid of zero entries so min() does not assign a zero value for the lower bound of length
                    end
                    
                    if ( BOUNDARYH_( z ) ~= 0 )
                        BOUNDARYH( p ) = BOUNDARYH_( z ); p = p + 1;
                    end                      
                end   
                
            end     
            a = 1; b = 1; q = 1; p = 1; % Reset accumulators.
            
            % Find length of digit existence per row by subtracting max()
            % and min() indexes across the sample.
            
            % Vertical lengths.

            [ v, ~ ] = max( BOUNDARYV ); [ x, ~ ] = min( BOUNDARYV );    
            
            if ( abs( v - x ) == 0 && v == 0 && x == 0 ) % Conditional for zero pixels, or a zero difference.
                 rowLengthV( n, i ) = rowLengthV( n, i ) + 0;
            elseif ( abs( v - x ) == 1 && v > x && x == 0 )  % Conditonal for one pixel or a unity difference.
                rowLengthV( n, i ) = rowLengthV( n, i ) + abs( v - x ); 
            else
                rowLengthV( n, i ) = rowLengthV( n, i ) + abs( v - x + 1 ); % Conditional to include boundary values for all other lengths.
            end
            
            % Horozontal lengths.
            
             [ v, ~ ] = max( BOUNDARYH ); [ x, ~ ] = min( BOUNDARYH );
            
            if ( abs( v - x ) == 0 && v == 0 && x == 0 )
                rowLengthH( n, i ) = rowLengthH( n, i ) + 0;                
            elseif ( abs( v - x ) == 1 && v > x && x == 0 )
                rowLengthH( n, i ) = rowLengthH( n, i ) + abs( v - x );
            else 
                rowLengthH( n, i ) = rowLengthH( n, i ) + abs( v - x + 1 );                
            end               
               
        end    
        BOUNDARYV = zeros( 1, 1 ); BOUNDARYH = zeros( 1, 1 );  
    end
    
    % Find total digit length by row...
    
    rowLengthsTotalV( n ) = sum( rowLengthV( n, : ), 2 );
    rowLengthsTotalH( n ) = sum( rowLengthH( n, : ), 2 );   
    
    % Generate cumulative PMFs of digit lengths per row for vertical and
    % horizontal projections...
    
    PMFLV( n, : ) = rowLengthV( n , : ) ./ rowLengthsTotalV( n );
    PMFLH( n, : ) = rowLengthH( n , : ) ./ rowLengthsTotalH( n );     
end
PMFLV = circshift( PMFLV, -1, 1 ); PMFLH = circshift( PMFLH, -1, 1 ); % Circular shift to place the PMFs of zero at index ten.

% Generate dissimilarity measure between all cumulative PMFs.

DP = abs( PMFV - PMFH );
DL = abs( PMFLV - PMFLH );

% figure(1);
% for i = 1:10
%  stem( D( i, : ), 'r' ); 
%  title( ' Cumulative Dissimilarity PMF ', i ); subplot( 5, 2, i ); 
%  ylabel('P[ pixel ]'); xlabel('Bin'); 
% end

% Reset temprorary storage variabes for use during classification....

 PMFV_ = zeros( 1, 28 ); PMFH_ = zeros( 1, 28 ); % Horizontal and vertical projection PMFs for a random sample. 
 
 PMFLV_ = zeros( 1, 28 ); PMFLH_ = zeros( 1, 28 ); % Horizontal and vertical length projection PMFs. 
 
 nzEntries = zeros( 28, ( totalDigits - trainingDigits ) ); 
 nzValues = zeros( 28, ( totalDigits - trainingDigits ) ); 
 
 nzEntries_ = zeros( 28, 1 ); 
 nzValues_ = zeros( 28, 1 );
 
 j = 1;
for i = ( trainingDigits + 1):totalDigits
    
    n = digitNumber( i ); % This is faster and less annoying.    
    
    % Make sure zero is at the bottom of the PMFs.
    if ( n == 0 )
        n = 10;
    end        
    
   DIGIT = reshape( dataset( i, 2 : 785 ), 28, 28 )'; 
   
   DIGIT = ( DIGIT > 127 ); % Binarization.
   
   sPMF = size( find( DIGIT( :, : ) ~= 0 ), 1 ); % Number of non-zero pixels.
   
   DIGITH = DIGIT; % Digit array for horizontal projections.
   
   DIGITV = permute( DIGIT, [ 2, 1 ] ); % Transpose to find vertical projections simulateously in dissimilarity().
   
   [ DP_, nzEntries_, nzValues_, PMFV_, PMFH_ ] = dissimilarity( DIGITV, DIGITH, sPMF ); % Finds dissimilarity PMF and the number of non-zero entries and their associated values in the dissimilarity PMF.        
    
    nzEntries( 1:length( nzEntries_ ), j ) = nzEntries_( 1, 1:end );
    nzValues( 1:length( nzValues_ ), j ) = nzValues_( 1, 1:end ); j = j + 1;   

    [ vP1, vP2, vP3, vP4, hP1, hP2, hP3, hP4 ] = maxNumPixelIndexes( PMFV_, PMFH_ );

    [ DL_, vL1, vL2, vL3, vL4, hL1, hL2, hL3, hL4 ] = maxLengthIndexes( DIGITV, DIGITH );      
    
    decidedDigit = bayesianClassifier( vP1, vP2, vP3, vP4, hP1, hP2, hP3, hP4, vL1, vL2, vL3, vL4, hL1, hL2, hL3, hL4 ); 
    
    DECISIONS = DECISIONS + 1;   
    
    if ( decidedDigit ~= n )
        ERRORS = ERRORS + 1;
    end    
    PERROR0 = ERRORS / DECISIONS
end
PERROR = ERRORS / DECISIONS; toc


% Use the dissimilarity measure to find the difference between vertical and
% horizontal PMFs...

function  [ DP_, nzEntries_, nzValues_, PMFV_, PMFH_ ] = dissimilarity( DIGITV, DIGITH, sPMF )    
    global PMFV_ PMFH_ DP_ nzEntries_ nzValues_

    for i = 1:28
       PMFV_( i ) = size( find( DIGITV( :, i ) ~= 0 ), 1 );
       PMFH_( i ) = size( find( DIGITH( :, i ) ~= 0 ), 1 );
    end
    PMFV_ = PMFV_ / sPMF ;
    PMFH_ = PMFH_ / sPMF ;
    
    DP_ = abs( PMFV_ - PMFH_ ); 
    
    [ ~, v, w ] =  find( DP_ ); nzEntries_ = v; nzValues_ = w;          
end 

% FEATURE #1: Ascending maximim pixel count per digit per projection
% direction....

function [ vP1, vP2, vP3, vP4, hP1, hP2, hP3, hP4 ] = maxNumPixelIndexes( PMFV_, PMFH_ )
    global PMFV_ PMFH_

    for i = 1
        for j = 1:4    
            [ maxPixelsAmplitudesV( i, j ), maxPixelIndexesV( i, j ) ] = max( PMFV_( i, : ) ); 
            PMFV_( i, maxPixelIndexesV( i, j ) ) = 0;    
             [ maxPixelsAmplitudesH( i, j ), maxPixelIndexesH( i, j ) ] = max( PMFH_( i, : ) ); 
            PMFH_( i, maxPixelIndexesH( i, j ) ) = 0;    
        end
    end   
    
    vP1 = maxPixelIndexesV( 1, 1 ); vP2 = maxPixelIndexesV( 1, 2 );
    vP3 = maxPixelIndexesV( 1, 3 ); vP4 = maxPixelIndexesV( 1, 4 );        
    
    hP1 = maxPixelIndexesH( 1, 1 ); hP2 = maxPixelIndexesH( 1, 2 );
    hP3 = maxPixelIndexesH( 1, 3 ); hP4 = maxPixelIndexesH( 1, 4 );     
end

% FEATURE #2: Ascending maximim lengths per digit per projection
% direction....

function [ DL_, vL1, vL2, vL3, vL4, hL1, hL2, hL3, hL4 ] = maxLengthIndexes( DIGITV, DIGITH )
    global PMFLV_ PMFLH_ DL_
    
    BOUNDARYV_ = zeros( 28,1 ); BOUNDARYH_ = zeros( 28,1 ); 
    
    BOUNDARYV = zeros( 1,1 ); BOUNDARYH = zeros( 1,1 ); 

    rowLengthV = zeros( 1, 28 ); rowLengthH = zeros( 1, 28 );
    
     a = 1; b = 1; q = 1; p = 1;    
     for i = 1:28  
         
        for j = 1:28

            % Find digit length per row. Locate where digit existence begins and ends.

            if ( DIGITV( i, j ) == 1 ) 
                BOUNDARYV_( a ) = j; a = a + 1;
            elseif ( DIGITH( i, j ) == 1 )
                BOUNDARYH_( b ) = j; b = b + 1;
            end     

            for z = 1:28  
                
                if( BOUNDARYV_( z ) ~= 0 )                    
                    BOUNDARYV( q ) = BOUNDARYV_( z ); q = q + 1; % Get rid of zero entries so min() does not assign a zero value for the lower bound of length
                end
                
                if ( BOUNDARYH_( z ) ~= 0 )
                    BOUNDARYH( p ) = BOUNDARYH_( z ); p = p + 1;
                end  
            end                

        end     
        a = 1; b = 1; q = 1; p = 1;

        % Find length of digit existence per row by subtracting max()
        % and min() indexes across the sample.

        % Vertical lengths.

        [ v, ~ ] = max( BOUNDARYV ); [ x, ~ ] = min( BOUNDARYV ); 

        if ( abs( v - x ) == 0 && v == 0 && x == 0 ) % Conditional for zero pixels, or a zero difference.
             rowLengthV( 1, i ) = rowLengthV( 1, i ) + 0;
        elseif ( abs( v - x ) == 1 && v > x && x == 0 ) % Conditonal for one pixel or a unity difference.
            rowLengthV( 1, i ) = rowLengthV( 1, i ) + abs( v - x ); 
        else
            rowLengthV( 1, i ) = rowLengthV( 1, i ) + abs( v - x + 1 ); % Conditional to include boundary values for all other lengths.
        end

        % Horizontal lengths.

         [ v, ~ ] = max( BOUNDARYH ); [ x, ~ ] = min( BOUNDARYH );

        if ( abs( v - x ) == 0 && v == 0 && x == 0 )
            rowLengthH( 1, i ) = rowLengthH( 1, i ) + 0;                
        elseif ( abs( v - x ) == 1 && v > x && x == 0 )
            rowLengthH( 1, i ) = rowLengthH( 1, i ) + abs( v - x );
        else 
            rowLengthH( 1, i ) = rowLengthH( 1, i ) + abs( v - x + 1 ); % Include boundary values of the length.
        end
        
     end
    
    % Find total digit length by row...
    
    rowLengthsTotalV = sum( rowLengthV( : ), 2 );
    rowLengthsTotalH = sum( rowLengthH( : ), 2 );   
    
    % Generate cumulative PMFs of digit lengths per row for vertical and
    % horizontal projections for the random sample...
    
    PMFLV_( 1, : ) = rowLengthV( : ) ./ rowLengthsTotalV;
    PMFLH_( 1, : ) = rowLengthH( : ) ./ rowLengthsTotalH;    
    
    DL_ = abs( PMFLV_ - PMFLH_ ); 
    
     for i = 1
        for j = 1:4    
            [ maxLengthMagnitudesV( i, j ), maxLengthIndexesV( i, j ) ] = max( PMFLV_( i, : ) ); 
            PMFLV_( i, maxLengthIndexesV( i, j ) ) = 0;    
             [ maxLengthMagnitudesH( i, j ), maxLengthIndexesH( i, j ) ] = max( PMFLH_( i, : ) ); 
            PMFLH_( i, maxLengthIndexesH( i, j ) ) = 0;    
        end
    end   
    
    vL1 = maxLengthIndexesV( 1, 1 ); vL2 = maxLengthIndexesV( 1, 2 );
    vL3 = maxLengthIndexesV( 1, 3 ); vL4 = maxLengthIndexesV( 1, 4 );        
    
    hL1 = maxLengthIndexesH( 1, 1 ); hL2 = maxLengthIndexesH( 1, 2 );
    hL3 = maxLengthIndexesH( 1, 3 ); hL4 = maxLengthIndexesH( 1, 4 );  
    
end   

% Implement Bayesian Classifier with FEATURE #1 and FEATURE # 2...

function decidedDigit = bayesianClassifier( vP1, vP2, vP3, vP4, hP1, hP2, hP3, hP4, vL1, vL2, vL3, vL4, hL1, hL2, hL3, hL4 )  
    global PMFV PMFH PMFLV PMFLH

    vP = [  vP1 vP2 vP3 vP4 ]; hP = [ hP1 hP2 hP3 hP4 ]; vL = [ vP1 vP2 vP3 vP4 ]; hL = [ hP1 hP2 hP3 hP4 ];
    
% vP = [  vP1 vP2 vP3 vP4 ]; hP = [ hP1 hP2 hP3 hP4 ]; vL = [ vL1 vL2 vL3 vL4 ]; hL = [ hL1 hL2 hL3 hL4 ];
    
%     vPR = randi( vP ); hPR = randi( hP ); vLR = randi( vL ); hLR = randi( hL );

%     for z = 1:4
%         for j = 1:10
%             for i = 1:10        
%                 if( PMFV( i, vPR( z ) ) * PMFH( i, hPR( z ) ) * PMFLV( i, vLR( z ) ) * PMFLH( i, hLR( z ) ) > PMFV( 1, vPR( z ) ) * PMFH( 1, hPR( z ) ) * PMFLV( 1, vLR( z )  ) * PMFLH( 1, hLR( z ) ) )            
%                     decidedDigit( z ) = i;            
%                 end             
%             end
%         end        
%     end
  
for z = 1
    for j = 1
        for i = 1:10        
            if ( PMFV( i, vP( z ) ) * PMFH( i, hP( z ) ) * PMFLV( i, vL( z ) ) * PMFLH( i, hL( z ) ) > PMFV( j, vP( z ) ) * PMFH( j, hP( z ) ) * PMFLV( j, vL( z )  ) * PMFLH( j, hL( z ) ) )            
                decidedDigit = i; 
            end             
        end
    end
end       
    
%     for z = 1:4
%         for j = 1:10
%             for i = 1:10        
%                 if ( PMFV( i, vP( z ) ) * PMFH( i, hP( z ) ) > PMFV( j, vP( z ) ) * PMFH( j, hP( z ) ) )            
%                     decidedDigit = i;
%                 end             
%             end
%         end
%     end       

%     for z = 1:4
%         for j = 1:10
%             for i = 1:10        
%                 if ( PMFLV( i, vL( z ) ) * PMFLH( i, hL( z ) ) > PMFLV( 1, vL( z )  ) * PMFLH( 1, hL( z ) ) )            
%                     decidedDigit = i; 
%                 end             
%             end
%         end
%     end   
     
% for z = 1:4
%     for j = 1:10
%         for i = 1:10        
%             if( PMFV( i, vP( z ) ) * PMFH( i, hP( z ) ) > PMFV( j, vP( z ) ) * PMFH( j, hP( z ) ) && PMFLV( i, vL( z ) ) * PMFLH( i, hL( z ) ) > PMFLV( j, vL( z )  ) * PMFLH( j, hL( z ) ) )            
%                 decidedDigit = i;
%             end             
%         end
%     end
% end      
    
end