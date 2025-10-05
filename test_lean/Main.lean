-- import TestLean

-- def main : IO Unit :=
--   IO.println s!"Hello, {hello}!"

-- def lean : String := "Lean"
-- def hello := "Hello"
-- #check String.append hello (String.append " " lean)

def add1 (n : Nat) : Nat := n + 1
#eval add1 7

-- 1.4 Structures
structure Point where
  x : Float
  y : Float
  deriving Repr

def origin : Point := { x := 0.0, y := 0.0 }
#eval origin

-- Can't use this syntax because the type is not known
--#check { x:= 0.0, y := 0.0 }

#check ({ x:= 0.0, y := 0.0 } : Point)
-- Syntattical sugar.
#check { x:= 0.0, y := 0.0 : Point}

def addPoints( p1 : Point) (p2 : Point ) : Point :=
  { x := p1.x + p2.x, y := p1.y + p2.y }

#eval addPoints { x := 1.0, y := 2.0 } { x := 3.0, y := 4.0 }

-- Constructor
#eval Point.mk 1.5 2.8
#check (Point.mk)
#eval ({ x := 1.5, y := 2.8 } : Point)

-- Accessor
#check (Point.x)
#eval origin.x

-- dot notation on functions

-- 1.5 Types

inductive Bool_ where
  | false_ : Bool_
  | true_ : Bool_

#eval Bool_.false_
#eval Bool_.true_

#check Bool_.false_
#check Bool_.true_

--
inductive Nat_ where
  | zero : Nat_
  -- Constructor takes an argument.
  | succ (n : Nat_) : Nat_

--
#check Nat_.zero
#check Nat_.succ
#eval Nat_.succ (Nat_.succ (Nat_.zero))

-- 1.6 Pattern matching

def isZero (n : Nat_) : Bool :=
  match n with
  | Nat_.zero => true
  | Nat_.succ k => false

#eval isZero Nat_.zero
#eval isZero (Nat_.succ (Nat_.zero))

-- 1.7 Recursive functions

def even (n : Nat_) : Bool  :=
  match n with
  -- Base cases.
  | Nat_.zero => true
  -- Structural recursion.
  | Nat_.succ k => not (even k)

#eval even Nat_.zero
#eval even (Nat_.succ (Nat_.zero))

-- 1.8 Polymorphic types

structure PPoint (type: Type) where
  x : type
  y : type
deriving Repr

def natOrigin: PPoint Nat :=
  { x := Nat.zero, y:= Nat.zero }

def floatOrigin: PPoint Float :=
  { x := 0.0, y:= 0.0 }

#eval natOrigin
#eval floatOrigin

def replaceX (type: Type) (point : PPoint type) (newX : type) : PPoint type :=
  { point with x := newX }

#check (replaceX)
#check replaceX Nat
#check replaceX Nat natOrigin
#check replaceX Nat natOrigin 5

-- 1.6.1 Lists

inductive List_ (type: Type) where
  -- Empty list.
  | nil: List_ type
  | cons: type -> List_ type -> List_ type

#check (List_.nil)
#check (List_.cons)
#eval List_.cons 1 (List_.cons 2 (List_.nil))

def explicitPrimesUnder10 : List_ Nat :=
  List_.cons 2 (List_.cons 3 (List_.cons 5 (List_.cons 7 (List_.nil))))

#eval explicitPrimesUnder10

def primesUnder10 : List Nat :=
  List.cons 2 (List.cons 3 (List.cons 5 (List.cons 7 (List.nil))))

#eval primesUnder10
