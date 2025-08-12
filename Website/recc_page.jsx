"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Checkbox } from "@/components/ui/checkbox"
import { Sparkles } from "lucide-react"

const brandOptions = ["Lululemon", "Zara", "Uniqlo", "H&M", "Nike", "Adidas", "Gap", "Forever 21"]
const styleOptions = ["casual chic", "athleisure", "bohemian", "minimalist", "vintage", "streetwear", "formal", "romantic"]
const colorOptions = ["soft navy", "cream", "dusty rose", "sage green", "charcoal", "blush pink", "camel", "ivory"]
const materialOptions = ["modal", "linen", "cotton blend", "silk", "wool", "polyester", "bamboo", "cashmere"]
const fitOptions = ["slim fit", "relaxed fit", "oversized", "tailored", "regular fit"]
const colorToneOptions = ["neutral", "warm", "cool", "bold"]
const bodyShapeOptions = ["hourglass", "pear", "apple", "rectangle", "inverted triangle"]

const sampleClothingItems = [
  {
    id: 1,
    name: "T-Shirt",
    brand: "Nike",
    style: "casual chic",
    color: "soft navy",
    material: "cotton blend",
    fit: "regular fit",
  },
  { id: 2, name: "Jeans", brand: "Levi's", style: "athleisure", color: "cream", material: "denim", fit: "slim fit" },
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
    setTimeout(() => {
      setRecommendations(sampleClothingItems)
      setIsLoading(false)
    }, 1500)
  }

  return (
    <div className="min-h-screen bg-amber-50">
      <div className="container mx-auto px-4 py-8">
        <div className="w-full">
          {/* Left Side - User Preferences Form */}
          <div className="bg-transparent p-8 w-full max-w-none">
            <div className="flex items-center gap-3 mb-8">
              <Sparkles className="w-8 h-8 text-[#a00a11]" />
              <h1 className="text-3xl font-bold text-[#a00a11]">Get to know you</h1>
            </div>

            <div className="space-y-10">
              {/* Preferred Brands */}
              <div className="mb-8">
                <Label className="text-lg font-semibold text-[#a00a11] mb-3 block">Preferred Brands</Label>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4 mb-8">
                  {brandOptions.map((brand) => (
                    <div key={brand} className="flex items-center space-x-2">
                      <Checkbox
                        id={`brand-${brand}`}
                        checked={preferences.preferred_brands.includes(brand)}
                        onCheckedChange={(checked) => handleCheckboxChange("preferred_brands", brand, checked)}
                        className="border-red-300 data-[state=checked]:bg-red-500"
                      />
                      <Label htmlFor={`brand-${brand}`} className="text-sm text-[#a00a11]">
                        {brand}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Preferred Styles */}
              <div className="mb-8">
                <Label className="text-lg font-semibold text-[#a00a11] mb-3 block">Preferred Styles</Label>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4 mb-8">
                  {styleOptions.map((style) => (
                    <div key={style} className="flex items-center space-x-2">
                      <Checkbox
                        id={`style-${style}`}
                        checked={preferences.preferred_styles.includes(style)}
                        onCheckedChange={(checked) => handleCheckboxChange("preferred_styles", style, checked)}
                        className="border-red-300 data-[state=checked]:bg-red-500"
                      />
                      <Label htmlFor={`style-${style}`} className="text-sm text-[#a00a11] capitalize">
                        {style}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Preferred Colors */}
              <div className="mb-8">
                <Label className="text-lg font-semibold text-[#a00a11] mb-3 block">Preferred Colors</Label>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4 mb-8">
                  {colorOptions.map((color) => (
                    <div key={color} className="flex items-center space-x-2">
                      <Checkbox
                        id={`color-${color}`}
                        checked={preferences.preferred_colors.includes(color)}
                        onCheckedChange={(checked) => handleCheckboxChange("preferred_colors", color, checked)}
                        className="border-red-300 data-[state=checked]:bg-red-500"
                      />
                      <Label htmlFor={`color-${color}`} className="text-sm text-[#a00a11] capitalize">
                        {color}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Preferred Materials */}
              <div className="mb-8">
                <Label className="text-lg font-semibold text-[#a00a11] mb-3 block">Preferred Materials</Label>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4 mb-8">
                  {materialOptions.map((material) => (
                    <div key={material} className="flex items-center space-x-2">
                      <Checkbox
                        id={`material-${material}`}
                        checked={preferences.preferred_materials.includes(material)}
                        onCheckedChange={(checked) => handleCheckboxChange("preferred_materials", material, checked)}
                        className="border-red-300 data-[state=checked]:bg-red-500"
                      />
                      <Label htmlFor={`material-${material}`} className="text-sm text-[#a00a11] capitalize">
                        {material}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Preferred Fit */}
              <div className="mb-8">
                <Label className="text-lg font-semibold text-[#a00a11] mb-3 block">Preferred Fit</Label>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4 mb-8">
                  {fitOptions.map((fit) => (
                    <div key={fit} className="flex items-center space-x-2">
                      <Checkbox
                        id={`fit-${fit}`}
                        checked={preferences.preferred_fit.includes(fit)}
                        onCheckedChange={(checked) => handleCheckboxChange("preferred_fit", fit, checked)}
                        className="border-red-300 data-[state=checked]:bg-red-500"
                      />
                      <Label htmlFor={`fit-${fit}`} className="text-sm text-[#a00a11] capitalize">
                        {fit}
                      </Label>
                    </div>
                  ))}
                </div>
              </div>

              {/* Color Tone */}
              <div className="mb-8">
                <Label className="text-lg font-semibold text-[#a00a11] mb-3 block">Color Tone</Label>
                <Select
                  value={preferences.color_tone}
                  onValueChange={(value) => handleSelectChange("color_tone", value)}
                  className="w-full max-w-xs"
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
              <div className="mb-8">
                <Label className="text-lg font-semibold text-[#a00a11] mb-3 block">Body Shape</Label>
                <Select
                  value={preferences.body_shape}
                  onValueChange={(value) => handleSelectChange("body_shape", value)}
                  className="w-full max-w-xs"
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
                className="w-full bg-[#a00a11] hover:bg-red-600 text-white py-6 text-lg font-semibold rounded-xl"
              >
                {isLoading ? "Getting Recommendations..." : "Get Recommendations"}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
