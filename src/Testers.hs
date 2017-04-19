module Testers
( showDIMACS, showCNF, testResult, testRippleCarryDIMACS, solnRippleCarry, rippleCarryAsBsCin, rippleCarryAsBsCinList
, andCNF, testHalfAdderDIMACS, testFullAdderDIMACS, solnFullAdder, computeSolnFullAdder, rippleCarry
, popCountCompute, popCountLayer, popCount, popCountDIMACS, exhaust-- "exploration"
)
where

import Data.List -- for zip4
import Data.Tuple.Select
import Text.Read (readMaybe)
import Control.Monad
import Compiler



showDIMACS :: CNF -> Int -> String
showDIMACS cnf nVars = "p cnf " ++ show nVars ++ " " ++ show (length cnf)
  ++ "\n" ++ showCNF cnf

showCNF :: CNF -> String
showCNF = foldl(\acc andClause -> acc ++ -- bizarre head/tail splitting because want no leading space
  foldl (\acc or1 -> acc ++ " " ++ show or1) (show $ head andClause) (tail andClause) ++ " 0\n") ""


--------- Testing ! ------------------------------------------------------------


testHalfAdderDIMACS :: [String]
testHalfAdderDIMACS = map (`showDIMACS` 4) testHalfAdderConstraints
  where testHalfAdderConstraints :: [CNF]
        testHalfAdderConstraints = map (\x-> adderConstraints ++ andCNF [head x] ++ andCNF [(head . tail) x]) allInputs
          where (adderConstraints, _, _) = halfAdder 1 2 2 []
                allInputs = sequence [[1, -1], [2, -2]] -- 0+0, 0+1, 1+0, 1+1
----



testFullAdderDIMACS :: [String]
testFullAdderDIMACS = map (`showDIMACS` 5) testFullAdderConstraints
  where testFullAdderConstraints :: [CNF]
        testFullAdderConstraints = map (\x-> adderConstraints ++ andCNF (fstX x) ++ andCNF (sndX x) ++ andCNF (thdX x)) allInputs
          where (adderConstraints, _, _) = fullAdder 1 2 3 3 [] -- a b c #vars accum
                allInputs = sequence [[1, -1], [2, -2], [3, -3]] -- generates all 8 input combos (in counting order)
                fstX x = [head x] -- these tease apart the above tuples so we can "and" them as assertions
                sndX x = [x !! 1]
                thdX x = [x !! 2] -- now easier by index :)



-- s SATISFIABLE
-- v a b c_in c s 0
solnFullAdder :: [String]                        -- this formats the list to be space sep'd
solnFullAdder = map (\x -> "s SATISFIABLE\nv " ++ tail (foldl (\acc x-> acc ++ " " ++ show x) "" (computeSolnFullAdder x 4 5)) ++ " 0\n") allInputs
  where allInputs = sequence [[1, -1], [2, -2], [3, -3]] -- generates all 8 input combos (in counting order)


-- sum is positive iff (a+b+c) is odd
-- carry is positive iff (a+b+c) > 2
--                  [a, b, c] -> "a b c_in carry sum"
computeSolnFullAdder :: [Int] -> Int -> Int -> [Int]
computeSolnFullAdder incoming cindex sindex = incoming ++ [c] ++ [s]
  where total = sum $ map (\x -> if x < 0 then 0 else 1) incoming
        c = if total > 1 then cindex else (- cindex)
        s = if odd total then sindex else (- sindex)
-----------

-- RIPPLE CARRY !
testRippleCarryDIMACS :: Int -> [String]
testRippleCarryDIMACS numDigs = map (`showDIMACS` numVars) testRippleCarryConstraints
  where numVars = 1 + (4*numDigs)
        testRippleCarryConstraints :: [CNF]
        testRippleCarryConstraints = map (foldl (\acc y -> acc ++ andCNF [y]) adderConstraints) allInputs
          where (as, bs, cin) = rippleCarryAsBsCin numDigs
                (adderConstraints, _, _) = rippleCarry as bs cin cin [] -- [as] [bs] cin #vars accum
                allInputs = mapM (\x -> [x, -x]) [1..cin] -- generates all input combos (in counting order)


solnRippleCarry :: Int -> [String]            -- tail gets rid of a leading space
solnRippleCarry numDigs = map (\x -> "s SATISFIABLE\nv " ++ tail (foldl (\acc y-> acc ++ " " ++ show y) "" x) ++ " 0\n") result
  where numVars = 1 + (4*numDigs)
        allInputs = map rippleCarryAsBsCinList $ mapM (\x -> [x, -x]) [1..(2*numDigs + 1)]
        (as_in, bs_in, cin_in) = rippleCarryAsBsCin numDigs
        (_, cs, ss) = rippleCarry as_in bs_in cin_in cin_in [] -- [as] [bs] cin #vars accum
        cases = map (\(as, bs, cin) -> zip5 as bs (repeat cin) cs ss) allInputs
        result = map (\x -> go x (sel3 $ head x) []) cases
        -- dont ask questions :(      -- this is the initial cin, need this bc chaining
        go :: [(Int, Int, Int, Int, Int)] -> Int -> [Int] -> [Int]
        go []    _      accum = sortBy (\x y -> compare (abs x) (abs y)) accum
        go todo cin accum = go (tail todo) cout_val $ nub (accum ++ res)
          where (a, b, _, cout_index, s_index) = head todo
                res = computeSolnFullAdder [a, b, cin] cout_index s_index
                cout_val = res !! 3


rippleCarryAsBsCin :: Int -> ([Int], [Int], Int)
rippleCarryAsBsCin numDigs = (as, bs, cin)
  where as = take numDigs [1..]
        bs = drop numDigs $ take (2*numDigs) [1..]
        cin = 2*numDigs + 1


rippleCarryAsBsCinList :: [Int] -> ([Int], [Int], Int)
rippleCarryAsBsCinList inputList = (as, bs, cin)
  where numDigs = quot (length inputList) 2
        as = take numDigs inputList
        bs = drop numDigs $ take (2*numDigs) inputList
        cin = head $ drop (2*numDigs) inputList


----------------
-- Pop Count!
popCountDIMACS :: Int -> [String]
popCountDIMACS numDigs = map ((\ x -> showDIMACS (x ++ cnf) (maximum vars)) . map (: [])) allInputs
  where (cnf, vars) = popCount [1.. numDigs]
        allInputs = exhaust [1.. numDigs]

exhaust :: [Int] -> [[Int]]
exhaust [] = []
exhaust [x] = [[x], [-x]]
exhaust (x:xs) = concatMap (\ys -> [x:ys, (-x):ys]) (exhaust xs)




--------- Testing ! ------------------------------------------------------------
-- result <- readFile "popCountResults/popCounter3_1.sol"
-- result <- readFile "popCountResults/popCounter1_0.sol"
-- result <- readFile "popCountResults/underconstrained_9.sol"
data SATResult = Correct | Unsatisfiable | WrongResult Int Int | ParseError deriving(Show)



testResult :: String -> Int -> SATResult
testResult result setVars
  | numLines == 1 = Unsatisfiable
  | otherwise = correct
  where numLines = length $ lines result
        resVars = snd $ popCount [1.. setVars]
        inList = mapM readMaybe . init . concatMap (words . tail) . tail . lines $ result :: Maybe [Int]
        correct = case inList of
          Nothing -> ParseError
          Just x -> correctHuh x setVars resVars


correctHuh :: [Int] -> Int -> [Int] -> SATResult
correctHuh inList setVars resVars
  | nSetBits == resSetBits = Correct
  | otherwise = WrongResult nSetBits resSetBits
  where nSetBits = sum $ map (\x -> if x < 0 then 0 else 1) $ take setVars inList
        resBools = map ((> 0) . (inList !!) . subtract 1) resVars
        resSetBits = foldl (\acc bit -> if bit then acc*2+1 else acc*2) 0 resBools