"use client"

import { useState } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Sparkles, Search } from "lucide-react"

const brandOptions = ["Lululemon", "Zara", "Uniqlo", "H&M", "Nike", "Adidas", "Gap", "Forever 21"]
const styleOptions = [
  "casual chic",
  "athleisure",
  "bohemian",
  "minimalist",
  "vintage",
  "streetwear",
  "formal",
  "romantic",
]
const colorOptions = ["soft navy", "cream", "dusty rose", "sage green", "charcoal", "blush pink", "camel", "ivory"]
const materialOptions = ["modal", "linen", "cotton blend", "silk", "wool", "polyester", "bamboo", "cashmere"]
const fitOptions = ["slim fit", "relaxed fit", "oversized", "tailored", "regular fit"]
const colorToneOptions = ["neutral", "warm", "cool", "bold"]
const bodyShapeOptions = ["hourglass", "pear", "apple", "rectangle", "inverted triangle"]

const sampleClothingItems = [
  {
    id: 1,
    name: "Relaxed Linen Blouse",
    brand: "Zara",
    price: "$49.99",
    image: "/placeholder.svg?height=300&width=300",
  },
  {
    id: 2,
    name: "High-Waisted Joggers",
    brand: "Lululemon",
    price: "$128.00",
    image: "/placeholder.svg?height=300&width=300",
  },
  {
    id: 3,
    name: "Bohemian Maxi Dress",
    brand: "Uniqlo",
    price: "$79.99",
    image: "/placeholder.svg?height=300&width=300",
  },
  { id: 4, name: "Modal T-Shirt", brand: "Uniqlo", price: "$19.99", image: "/placeholder.svg?height=300&width=300" },
  { id: 5, name: "Casual Chic Blazer", brand: "Zara", price: "$89.99", image: "/placeholder.svg?height=300&width=300" },
  {
    id: 6,
    name: "Athleisure Set",
    brand: "Lululemon",
    price: "$158.00",
    image: "/placeholder.svg?height=300&width=300",
  },
]

export default function FashionRecommendation() {
  const [preferences, setPreferences] = useState({
    preferred_brands: [],
    preferred_styles: [],
    preferred_colors: [],
    preferred_materials: [],
    preferred_fit: [],
    color_tone: "",
    body_shape: "",
  })

  const [prompt, setPrompt] = useState("")
  const [recommendations, setRecommendations] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const handleCheckboxChange = (field, value, checked) => {
    setPreferences((prev) => ({
      ...prev,
      [field]: checked ? [...prev[field], value] : prev[field].filter((item) => item !== value),
    }))
  }

  const handleSelectChange = (field, value) => {
    setPreferences((prev) => ({
      ...prev,
      [field]: value,
    }))
  }

  const handleGetRecommendations = () => {
    setIsLoading(true)
    // Simulate API call
    setTimeout(() => {
      setRecommendations(sampleClothingItems)
      setIsLoading(false)
    }, 1500)
  }

  const handleVirtualTryOn = (item) => {
    console.log("Virtual try-on for:", item.name)
    // Implement virtual try-on functionality
  }

  const handleFindSimilar = (item) => {
    console.log("Find similar to:", item.name)
    // Implement find similar functionality
  }

  return (
    <div className="min-h-screen bg-amber-50">
      <div className="container mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8 h-full">
          {/* Left Side - User Preferences Form */}
          <div className="bg-white rounded-2xl shadow-lg p-8 h-fit">
            <div className="flex items-center gap-3 mb-8">
              <Sparkles className="w-8 h-8 text-red-500" />
              <h1 className="text-3xl font-bold text-gray-900">Get to know you</h1>
            </div>

            <div className="space-y-6">
              {/* Preferred Brands */}
              <div>
                <Label className="text-lg font-semibold text-gray-900 mb-3 block">Preferred Brands</Label>
                <div className="grid grid-cols-2 gap-2">
                  {brandOptions.map((brand) => (
                    <div key={brand} className="flex items-center space-x-2">
                      <Checkbox
                        id={`brand-${brand}`}
                        checked={preferences.preferred_brands.includes(brand)}
                        onCheckedChange={(checked) => handleCheckboxChange("preferred_brands", brand, checked)}
                        className="border-red-300 data-[state=checked]:bg-red-500"
                      />
                      <Label htmlFor={`brand-${brand}`} className="text-sm text-gray-700">
                        {brand}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Preferred Styles */}
              <div>
                <Label className="text-lg font-semibold text-gray-900 mb-3 block">Preferred Styles</Label>
                <div className="grid grid-cols-2 gap-2">
                  {styleOptions.map((style) => (
                    <div key={style} className="flex items-center space-x-2">
                      <Checkbox
                        id={`style-${style}`}
                        checked={preferences.preferred_styles.includes(style)}
                        onCheckedChange={(checked) => handleCheckboxChange("preferred_styles", style, checked)}
                        className="border-red-300 data-[state=checked]:bg-red-500"
                      />
                      <Label htmlFor={`style-${style}`} className="text-sm text-gray-700 capitalize">
                        {style}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Preferred Colors */}
              <div>
                <Label className="text-lg font-semibold text-gray-900 mb-3 block">Preferred Colors</Label>
                <div className="grid grid-cols-2 gap-2">
                  {colorOptions.map((color) => (
                    <div key={color} className="flex items-center space-x-2">
                      <Checkbox
                        id={`color-${color}`}
                        checked={preferences.preferred_colors.includes(color)}
                        onCheckedChange={(checked) => handleCheckboxChange("preferred_colors", color, checked)}
                        className="border-red-300 data-[state=checked]:bg-red-500"
                      />
                      <Label htmlFor={`color-${color}`} className="text-sm text-gray-700 capitalize">
                        {color}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Preferred Materials */}
              <div>
                <Label className="text-lg font-semibold text-gray-900 mb-3 block">Preferred Materials</Label>
                <div className="grid grid-cols-2 gap-2">
                  {materialOptions.map((material) => (
                    <div key={material} className="flex items-center space-x-2">
                      <Checkbox
                        id={`material-${material}`}
                        checked={preferences.preferred_materials.includes(material)}
                        onCheckedChange={(checked) => handleCheckboxChange("preferred_materials", material, checked)}
                        className="border-red-300 data-[state=checked]:bg-red-500"
                      />
                      <Label htmlFor={`material-${material}`} className="text-sm text-gray-700 capitalize">
                        {material}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Preferred Fit */}
              <div>
                <Label className="text-lg font-semibold text-gray-900 mb-3 block">Preferred Fit</Label>
                <div className="grid grid-cols-2 gap-2">
                  {fitOptions.map((fit) => (
                    <div key={fit} className="flex items-center space-x-2">
                      <Checkbox
                        id={`fit-${fit}`}
                        checked={preferences.preferred_fit.includes(fit)}
                        onCheckedChange={(checked) => handleCheckboxChange("preferred_fit", fit, checked)}
                        className="border-red-300 data-[state=checked]:bg-red-500"
                      />
                      <Label htmlFor={`fit-${fit}`} className="text-sm text-gray-700 capitalize">
                        {fit}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Color Tone */}
              <div>
                <Label className="text-lg font-semibold text-gray-900 mb-3 block">Color Tone</Label>
                <Select
                  value={preferences.color_tone}
                  onValueChange={(value) => handleSelectChange("color_tone", value)}
                >
                  <SelectTrigger className="border-red-200 focus:border-red-500">
                    <SelectValue placeholder="Select color tone" />
                  </SelectTrigger>
                  <SelectContent>
                    {colorToneOptions.map((tone) => (
                      <SelectItem key={tone} value={tone} className="capitalize">
                        {tone}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Body Shape */}
              <div>
                <Label className="text-lg font-semibold text-gray-900 mb-3 block">Body Shape</Label>
                <Select
                  value={preferences.body_shape}
                  onValueChange={(value) => handleSelectChange("body_shape", value)}
                >
                  <SelectTrigger className="border-red-200 focus:border-red-500">
                    <SelectValue placeholder="Select body shape" />
                  </SelectTrigger>
                  <SelectContent>
                    {bodyShapeOptions.map((shape) => (
                      <SelectItem key={shape} value={shape} className="capitalize">
                        {shape}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <Button
                onClick={handleGetRecommendations}
                disabled={isLoading}
                className="w-full bg-red-500 hover:bg-red-600 text-white py-6 text-lg font-semibold rounded-xl"
              >
                {isLoading ? "Getting Recommendations..." : "Get Recommendations"}
              </Button>
            </div>
          </div>

          {/* Right Side - Prompt and Recommendations */}
          <div className="space-y-6">
            {/* Prompt Section */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <div className="flex gap-4">
                <Textarea
                  placeholder="Describe what you're looking for... (e.g., 'I need a casual outfit for a weekend brunch')"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="flex-1 min-h-[100px] border-red-200 focus:border-red-500 resize-none"
                />
                <Button className="bg-red-500 hover:bg-red-600 text-white px-6">
                  <Search className="w-5 h-5" />
                </Button>
              </div>
            </div>

            {/* Recommendations Grid */}
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Recommended for You</h2>

              {recommendations.length === 0 ? (
                <div className="text-center py-12">
                  <div className="w-24 h-24 bg-amber-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Sparkles className="w-12 h-12 text-amber-600" />
                  </div>
                  <p className="text-gray-500 text-lg">
                    Fill out your preferences and get personalized recommendations!
                  </p>
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                  {recommendations.map((item) => (
                    <Card
                      key={item.id}
                      className="overflow-hidden border-2 border-amber-100 hover:border-red-200 transition-colors"
                    >
                      <CardContent className="p-0">
                        <div className="aspect-square relative">
                          <Image src={item.image || "/placeholder.svg"} alt={item.name} fill className="object-cover" />
                        </div>
                        <div className="p-4">
                          <div className="flex justify-between items-start mb-2">
                            <h3 className="font-semibold text-gray-900 text-sm">{item.name}</h3>
                            <Badge variant="secondary" className="bg-amber-100 text-amber-800">
                              {item.brand}
                            </Badge>
                          </div>
                          <p className="text-red-600 font-bold mb-4">{item.price}</p>
                          <div className="grid grid-cols-2 gap-2">
                            <Button
                              onClick={() => handleVirtualTryOn(item)}
                              className="bg-red-500 hover:bg-red-600 text-white text-sm py-2"
                            >
                              VITON
                            </Button>
                            <Button
                              onClick={() => handleFindSimilar(item)}
                              variant="outline"
                              className="border-red-200 text-red-600 hover:bg-red-50 text-sm py-2"
                            >
                              Find Similar
                            </Button>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
